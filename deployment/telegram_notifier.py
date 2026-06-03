"""
Telegram Notifier for Forecast Results

Sends formatted forecast messages to Telegram channel.
"""

import json
import os
import sys
import requests
from typing import Dict, List, Optional
from datetime import datetime

try:
    from deployment._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from utils.core.logger import get_logger
from utils.core.enums import TimeFrame

logger = get_logger(__name__)

# Bot credentials. Prop-firm bot is the legacy default; personal-account bot
# is a separate bot to keep the two signal streams in different channels.
_PROP_BOT_TOKEN = "8157808736:AAHhqYe9N_PQ4Ox2Khz-zMbKoytly9ugrGY"
_PROP_CHAT_ID = "-1002856645393"

# enigma_pa_notifications_bot posting to the "Enigma Signals - Personal
# Account" channel (chat_id resolved via getUpdates after adding the bot
# as channel admin). Overridable via env var.
_PERSONAL_BOT_TOKEN = "8698079967:AAEXjTkAJcHsh1B88E-dRa-YIQVLuQly6NE"
_PERSONAL_CHAT_ID = "-1003955204069"

# CFD prop-firm channel. Intentionally has NO hardcoded fallback so we can
# never accidentally post real-money CFD approvals to the wrong channel.
# Both values MUST come from the environment (TELEGRAM_CFD_PROP_BOT_TOKEN /
# TELEGRAM_CFD_PROP_CHAT_ID). When either is unset the notifier logs-only.
_CFD_PROP_BOT_TOKEN: Optional[str] = None
_CFD_PROP_CHAT_ID: Optional[str] = None


class TelegramNotifier:
    """
    Telegram bot for sending forecast notifications.

    Formats prediction results and sends them to a Telegram channel.
    Use :meth:`for_prop_firms` and :meth:`for_personal_account` classmethods
    to get a notifier bound to the correct bot/channel.
    """

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier.

        Parameters
        ----------
        token : str, optional
            Telegram bot token (if None, falls back to ``TELEGRAM_BOT_TOKEN``
            env var, then the prop-firm bot default)
        chat_id : str, optional
            Telegram chat ID (if None, falls back to ``TELEGRAM_CHAT_ID`` env
            var, then the prop-firm channel default)
        """
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", _PROP_BOT_TOKEN)
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", _PROP_CHAT_ID)

        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured. Messages will be logged only.")

        self.base_url = f"https://api.telegram.org/bot{self.token}"

    @classmethod
    def for_prop_firms(cls) -> "TelegramNotifier":
        """Notifier bound to the prop-firm signal channel (Enigma Notifications)."""
        token = os.environ.get("TELEGRAM_PROP_BOT_TOKEN", _PROP_BOT_TOKEN)
        chat_id = os.environ.get("TELEGRAM_PROP_CHAT_ID", _PROP_CHAT_ID)
        return cls(token=token, chat_id=chat_id)

    @classmethod
    def for_personal_account(cls) -> "TelegramNotifier":
        """Notifier bound to the personal-account signal channel (Enigma PA Notifications).

        The personal chat_id must be configured (via ``TELEGRAM_PERSONAL_CHAT_ID``
        env var or the ``_PERSONAL_CHAT_ID`` module constant). If unset, the
        notifier logs rather than sending so we never cross-post into the
        prop-firm channel by accident.
        """
        token = os.environ.get("TELEGRAM_PERSONAL_BOT_TOKEN", _PERSONAL_BOT_TOKEN)
        chat_id = os.environ.get("TELEGRAM_PERSONAL_CHAT_ID", _PERSONAL_CHAT_ID)
        instance = cls.__new__(cls)
        instance.token = token
        instance.chat_id = chat_id
        instance.base_url = f"https://api.telegram.org/bot{token}"
        if not chat_id:
            logger.warning(
                "Personal-account chat_id is unset; messages will be logged only. "
                "Set TELEGRAM_PERSONAL_CHAT_ID to enable sends."
            )
        return instance

    @classmethod
    def for_cfd_prop(cls) -> "TelegramNotifier":
        """Notifier bound to the CFD prop-firm signal channel.

        Unlike the prop-futures and personal-account notifiers, this one has
        **no hardcoded fallback credentials**. Both the bot token and chat id
        must come from the environment:

        - ``TELEGRAM_CFD_PROP_BOT_TOKEN``
        - ``TELEGRAM_CFD_PROP_CHAT_ID``

        If either is unset, the notifier logs rather than sending so we never
        cross-post CFD prop-firm execution approvals into another channel by
        accident, and the CFD execution orchestrator should refuse to request
        approval (fail closed) when this notifier is in logs-only mode.
        """
        token = os.environ.get("TELEGRAM_CFD_PROP_BOT_TOKEN", _CFD_PROP_BOT_TOKEN)
        chat_id = os.environ.get("TELEGRAM_CFD_PROP_CHAT_ID", _CFD_PROP_CHAT_ID)
        instance = cls.__new__(cls)
        instance.token = token
        instance.chat_id = chat_id
        instance.base_url = f"https://api.telegram.org/bot{token}" if token else ""
        if not token or not chat_id:
            logger.warning(
                "CFD prop-firm Telegram is not fully configured "
                "(token=%s, chat_id=%s). Messages will be logged only. "
                "Set TELEGRAM_CFD_PROP_BOT_TOKEN and TELEGRAM_CFD_PROP_CHAT_ID "
                "to enable sends.",
                "set" if token else "unset",
                "set" if chat_id else "unset",
            )
        return instance

    def is_configured(self) -> bool:
        """Return True iff both bot token and chat id are set (sends will go out)."""
        return bool(self.token) and bool(self.chat_id)
    
    def send_forecast_update(
        self, 
        forecasts: Dict[str, float], 
        timeframe: TimeFrame,
        timestamp: Optional[datetime] = None,
        market_status: Optional[str] = None
    ) -> bool:
        """
        Send forecast update message to Telegram.
        
        Parameters
        ----------
        forecasts : Dict[str, float]
            Dictionary of ticker -> forecast value (0-1)
        timeframe : TimeFrame
            Timeframe of the forecast (D or H4)
        timestamp : datetime, optional
            Timestamp of forecast (defaults to now)
        market_status : str, optional
            Market status message
            
        Returns
        -------
        bool
            True if message sent successfully
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format the message
        message = self._format_forecast_message(forecasts, timeframe, timestamp, market_status)
        
        # Send message
        return self.send_message(message)
    
    def _format_forecast_message(
        self, 
        forecasts: Dict[str, float], 
        timeframe: TimeFrame,
        timestamp: datetime,
        market_status: Optional[str] = None
    ) -> str:
        """
        Format forecast results into a Telegram message.
        
        Parameters
        ----------
        forecasts : Dict[str, float]
            Forecast results
        timeframe : TimeFrame
            Forecast timeframe
        timestamp : datetime
            Forecast timestamp
        market_status : str, optional
            Market status info
            
        Returns
        -------
        str
            Formatted message
        """
        # Header
        tf_name = "Daily" if timeframe == TimeFrame.D else "H4"
        message_lines = [
            "🤖 **FORECAST UPDATE**",
            f"📅 {tf_name} - {timestamp.strftime('%Y-%m-%d %H:%M EST')}"
        ]
        
        if market_status:
            message_lines.append(f"📊 Market: {market_status}")
        
        message_lines.append("")  # Empty line
        
        # Sort tickers for consistent display
        sorted_tickers = sorted(forecasts.keys())
        
        # Format each forecast
        for ticker in sorted_tickers:
            forecast = forecasts[ticker]
            emoji, sentiment = self._get_forecast_display(forecast, ticker)
            
            message_lines.append(f"{emoji} **{ticker}**: {forecast:.2f} _{sentiment}_")
        
        # Footer with next update info
        message_lines.append("")
        
        if timeframe == TimeFrame.D:
            message_lines.append("⏰ Next update: H4 forecast")
        else:
            # Calculate next H4 time
            next_h4_hours = [2, 6, 10, 14, 18, 22]
            current_hour = timestamp.hour
            next_hour = min([h for h in next_h4_hours if h > current_hour], default=next_h4_hours[0])
            if next_hour <= current_hour:
                next_hour = next_h4_hours[0]  # Next day
            message_lines.append(f"⏰ Next H4 update: {next_hour:02d}:00 EST")
        
        return "\n".join(message_lines)
    
    def _get_forecast_display(self, forecast: float, ticker: str) -> tuple:
        """
        Get emoji and sentiment text for forecast value.
        
        Parameters
        ----------
        forecast : float
            Forecast value (0-1)
        ticker : str
            Ticker symbol
            
        Returns
        -------
        tuple
            (emoji, sentiment_text)
        """
        # Choose emoji based on asset type
        if ticker in ['EURUSD', 'GBPUSD']:
            base_emoji = "💱"
        elif ticker in ['US500', 'US100']:
            base_emoji = "📈"
        else:
            base_emoji = "💰"
        
        # Determine sentiment
        if forecast >= 0.7:
            return f"{base_emoji}🟢", "Strong Bullish"
        elif forecast >= 0.6:
            return f"{base_emoji}🔵", "Bullish"
        elif forecast >= 0.4:
            return f"{base_emoji}⚪", "Neutral"
        elif forecast >= 0.3:
            return f"{base_emoji}🟠", "Bearish"
        else:
            return f"{base_emoji}🔴", "Strong Bearish"
    
    def send_message(self, text: str) -> bool:
        """
        Send a text message to Telegram.
        
        Parameters
        ----------
        text : str
            Message text (supports Markdown)
            
        Returns
        -------
        bool
            True if sent successfully
        """
        if not self.token or not self.chat_id:
            logger.info(f"Telegram not configured. Would send: {text}")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    def send_error_notification(self, error_message: str, component: str = "ForecastServer") -> bool:
        """
        Send error notification to Telegram.
        
        Parameters
        ----------
        error_message : str
            Error description
        component : str
            Component that generated the error
            
        Returns
        -------
        bool
            True if sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M EST')
        message = f"🚨 **FORECAST ERROR**\n📅 {timestamp}\n🔧 Component: {component}\n❌ Error: {error_message}"
        
        return self.send_message(message)
    
    def send_status_update(self, status: str, details: Optional[str] = None) -> bool:
        """
        Send status update to Telegram.
        
        Parameters
        ----------
        status : str
            Status message
        details : str, optional
            Additional details
            
        Returns
        -------
        bool
            True if sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M EST')
        message = f"ℹ️ **STATUS UPDATE**\n📅 {timestamp}\n📊 {status}"
        
        if details:
            message += f"\n📝 {details}"
        
        return self.send_message(message)
    
    def send_with_inline_keyboard(
        self,
        text: str,
        buttons: List[Dict[str, str]],
    ) -> Optional[int]:
        """Send a message with an inline keyboard. Returns message_id on success.

        Parameters
        ----------
        text
            Message body (Markdown).
        buttons
            Iterable of ``{"text": "...", "callback_data": "..."}`` dicts.
            One row, one button per entry.
        """
        if not self.token or not self.chat_id:
            logger.info(f"Telegram not configured. Would send: {text}")
            return None
        try:
            reply_markup = {"inline_keyboard": [[b] for b in buttons]}
            response = requests.post(
                f"{self.base_url}/sendMessage",
                data={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                    "reply_markup": json.dumps(reply_markup),
                },
                timeout=10,
            )
            if response.status_code != 200:
                logger.error(f"Telegram sendMessage error: {response.status_code} - {response.text}")
                return None
            return int(response.json()["result"]["message_id"])
        except Exception as e:
            logger.error(f"send_with_inline_keyboard failed: {e}")
            return None

    def send_with_inline_keyboard_rows(
        self,
        text: str,
        button_rows: List[List[Dict[str, str]]],
    ) -> Optional[int]:
        """Send a message with an inline keyboard, one button row per inner list.

        Used by :func:`execution.approval_flow.request_batch_approval` to lay
        out approve_all / cancel_all / per-account cancel rows distinctly.
        """
        if not self.token or not self.chat_id:
            logger.info(f"Telegram not configured. Would send: {text}")
            return None
        try:
            reply_markup = {"inline_keyboard": button_rows}
            response = requests.post(
                f"{self.base_url}/sendMessage",
                data={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                    "reply_markup": json.dumps(reply_markup),
                },
                timeout=10,
            )
            if response.status_code != 200:
                logger.error(
                    f"Telegram sendMessage (rows) error: {response.status_code} - {response.text}"
                )
                return None
            return int(response.json()["result"]["message_id"])
        except Exception as e:
            logger.error(f"send_with_inline_keyboard_rows failed: {e}")
            return None

    def edit_message_text(self, message_id: int, text: str) -> bool:
        """Edit a previously-sent message, clearing its inline keyboard."""
        if not self.token or not self.chat_id:
            return False
        try:
            response = requests.post(
                f"{self.base_url}/editMessageText",
                data={
                    "chat_id": self.chat_id,
                    "message_id": message_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"edit_message_text failed: {e}")
            return False

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> bool:
        """Acknowledge a callback_query so Telegram stops showing the spinner."""
        if not self.token:
            return False
        try:
            response = requests.post(
                f"{self.base_url}/answerCallbackQuery",
                data={"callback_query_id": callback_query_id, "text": text},
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"answer_callback_query failed: {e}")
            return False

    def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: int = 25,
        allowed_updates: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Long-poll getUpdates. Returns the raw ``result`` list on success."""
        if not self.token:
            return []
        try:
            params: Dict[str, object] = {"timeout": timeout}
            if offset is not None:
                params["offset"] = offset
            if allowed_updates is not None:
                params["allowed_updates"] = json.dumps(allowed_updates)
            # Network timeout must exceed the long-poll timeout.
            response = requests.get(
                f"{self.base_url}/getUpdates",
                params=params,
                timeout=timeout + 5,
            )
            if response.status_code != 200:
                logger.error(f"getUpdates error: {response.status_code} - {response.text}")
                return []
            return response.json().get("result", [])
        except Exception as e:
            logger.error(f"get_updates failed: {e}")
            return []

    def test_connection(self) -> bool:
        """
        Test Telegram connection by sending a test message.
        
        Returns
        -------
        bool
            True if test successful
        """
        test_message = "🧪 **FORECAST SYSTEM TEST**\n✅ Telegram connection working"
        return self.send_message(test_message)

    def probe_health(self) -> bool:
        """Lightweight reachability probe used by approval flow.

        Calls ``getMe`` (no side effects, no message sent) and returns True
        iff the bot token is valid and the API is reachable. Used by
        :func:`execution.approval_flow.request_approval` and
        :func:`execution.approval_flow.request_batch_approval` to distinguish
        a real user-timeout from a Telegram outage when ``default_on_timeout``
        is set to ``APPROVED`` (fail-closed on outage).
        """
        if not self.token:
            return False
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=5)
            if response.status_code != 200:
                logger.warning(
                    f"Telegram health probe returned {response.status_code}: {response.text[:200]}"
                )
                return False
            payload = response.json()
            return bool(payload.get("ok", False))
        except requests.exceptions.RequestException as e:
            logger.warning(f"Telegram health probe network error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Telegram health probe failed: {e}")
            return False

# Utility function for easy access
def create_telegram_notifier() -> TelegramNotifier:
    """
    Create and return a Telegram notifier instance.
    
    Returns
    -------
    TelegramNotifier
        Configured Telegram notifier
    """
    return TelegramNotifier()
