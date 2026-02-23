"""
Telegram Notifier for Forecast Results

Sends formatted forecast messages to Telegram channel.
"""

import os
import sys
import requests
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.core.logger import get_logger
from utils.core.enums import TimeFrame

logger = get_logger(__name__)

class TelegramNotifier:
    """
    Telegram bot for sending forecast notifications.
    
    Formats prediction results and sends them to a Telegram channel.
    """
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier.
        
        Parameters
        ----------
        token : str, optional
            Telegram bot token (if None, gets from env var)
        chat_id : str, optional
            Telegram chat ID (if None, gets from env var)
        """
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "8157808736:AAHhqYe9N_PQ4Ox2Khz-zMbKoytly9ugrGY")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "-1002856645393")
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured. Messages will be logged only.")
        
        self.base_url = f"https://api.telegram.org/bot{self.token}"
    
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
