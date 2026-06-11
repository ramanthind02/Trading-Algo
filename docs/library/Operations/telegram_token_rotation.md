# Telegram Bot Token Rotation Guide

## Situation

Three Telegram bots are used to send alerts and approvals for trading operations:

| Bot | Purpose | Env var (token) | Env var (chat_id) |
|-----|---------|-------------------|-------------------|
| Futures Prop-Firm Bot | Signal notifications for prop-firm futures accounts | `TELEGRAM_PROP_BOT_TOKEN` | `TELEGRAM_PROP_CHAT_ID` |
| Personal Account Bot | Signal notifications for personal trading accounts | `TELEGRAM_PERSONAL_BOT_TOKEN` | `TELEGRAM_PERSONAL_CHAT_ID` |
| CFD Prop-Firm Bot | Execution approvals for CFD prop-firm accounts | `TELEGRAM_CFD_PROP_BOT_TOKEN` | `TELEGRAM_CFD_PROP_CHAT_ID` |

**Status**: The code in `lib/core/notify.py` is clean — all tokens are now environment variables with `None` fallbacks. No hardcoded token strings remain in the source tree.

## When to Rotate

Rotate tokens when:

1. **A token has been leaked or exposed** (e.g., in git history, logs, or an accidental screenshot)
2. **A bot has been compromised or shows unexpected activity** (e.g., messages from unknown users)
3. **As a periodic security practice** (recommended: quarterly or semiannually)

The current tokens **have been leaked in git history** and **must be rotated to be secure**. Old tokens in the git log remain valid until explicitly revoked via BotFather.

## Rotation Steps

### 1. Revoke Old Tokens via BotFather

For each of the three bots:

1. Open Telegram on your phone or desktop client
2. Find **@BotFather** (Telegram's official bot management service)
3. Send `/mybots`
4. Select the bot to revoke (e.g., `@quant_foundry_prop_firm_bot`)
5. Select **API Token**
6. Select **Revoke current token** (this immediately invalidates the old token)
7. Copy the **new token** generated (Telegram will display it once)
8. **Save the new token securely** (password manager recommended)

**Repeat step 1-8 for all three bots.**

### 2. Update Your Environment Variables

Update your local `.env` file with the new tokens:

```bash
# ── Futures prop-firm Telegram channel ──────────────────────────────────────
TELEGRAM_PROP_BOT_TOKEN=<new_token_from_botfather>
TELEGRAM_PROP_CHAT_ID=<your_channel_chat_id>

# ── Personal Telegram channel ────────────────────────────────────────────────
TELEGRAM_PERSONAL_BOT_TOKEN=<new_token_from_botfather>
TELEGRAM_PERSONAL_CHAT_ID=<your_channel_chat_id>

# ── CFD prop-firm Telegram channel ──────────────────────────────────────────
TELEGRAM_CFD_PROP_BOT_TOKEN=<new_token_from_botfather>
TELEGRAM_CFD_PROP_CHAT_ID=<your_channel_chat_id>
```

**Note**: Do NOT commit `.env` to git (it is gitignored). The `.env.example` file contains template placeholders only.

### 3. Restart Services

Restart any running services that send Telegram notifications:

```powershell
# Stop the forecast server (if running)
Ctrl+C  # or Stop-Process if running in background

# Restart the forecast server
.\.venv\Scripts\python.exe -m deployment.forecast_server

# Restart the live node (if running)
.\.venv\Scripts\python.exe -m deployment.live.run_vault_sandbox --broker ftmo --exec-tier demo
```

Services will automatically load the new tokens from `.env` on startup.

### 4. Verify the New Tokens Work

Run a quick connection test to each notifier:

```python
# From the repo root in a Python shell or script:
from lib.core.notify import TelegramNotifier

# Test each bot
notifier_prop = TelegramNotifier.for_prop_firms()
success_prop = notifier_prop.send_message("🧪 Futures bot token rotation test — ignore")
print(f"Prop bot: {'✅ Sent' if success_prop else '❌ Failed (check token/chat_id)'}")

notifier_personal = TelegramNotifier.for_personal_account()
success_personal = notifier_personal.send_message("🧪 Personal bot token rotation test — ignore")
print(f"Personal bot: {'✅ Sent' if success_personal else '❌ Failed (check token/chat_id)'}")

notifier_cfd = TelegramNotifier.for_cfd_prop()
success_cfd = notifier_cfd.send_message("🧪 CFD prop bot token rotation test — ignore")
print(f"CFD bot: {'✅ Sent' if success_cfd else '❌ Failed (check token/chat_id)'}")
```

If any test returns `❌ Failed`, verify:
- The new token is copied correctly (no extra spaces)
- The chat_id is correct (run `/getid` in your Telegram bot to confirm)
- The bot has write permissions in the channel
- The `.env` file is in the repo root (not in a subdirectory)

### 5. Verify in Production

Wait 5–10 minutes, then:
1. Run the forecast server or live node
2. Generate a forecast or manual alert
3. Confirm the message arrives in Telegram
4. Check for any error logs in the console

## Code References

- **Token loading**: `lib/core/notify.py` lines 25–35 (environment variable declarations)
- **Notifier classes**: `lib/core/notify.py` lines 69–128 (classmethods for each bot)
- **Example usage**: `lib/core/notify.py` lines 557–584 (CLI entrypoint)
- **Forecast server integration**: `deployment/forecast_server.py` (sends alerts via notifiers)

## FAQ

**Q: Do I need to update anything in the source code?**
No. The source code `lib/core/notify.py` already reads tokens from the environment. Just update `.env`.

**Q: What if I don't know my chat_id?**
Send `/getid` to **@get_id_bot** in Telegram (a public utility bot). It will reply with your numeric user/chat ID.

**Q: Can I rotate one bot at a time?**
Yes. Rotate them in any order, one per day if needed. There is no coordination required between bots.

**Q: What if a service fails to start after rotation?**
Check the logs for:
- `Telegram token or chat_id not configured` → Token or chat_id is missing or empty in `.env`
- `Telegram API error` → Token is invalid (verify it was copied correctly, or revoke and request a new one via BotFather)

**Q: How do I know if a bot is actually secured after rotation?**
The old token in git history is now useless — it has been revoked and cannot be used to send messages. The new token is only in your local `.env` file, which is gitignored and never committed.

## Further Reading

- [Telegram Bot API](https://core.telegram.org/bots/api) — official reference
- [BotFather Guide](https://core.telegram.org/bots/botfather) — token and webhook management
- `.env.example` — template for environment variables
- `lib/core/notify.py` — TelegramNotifier implementation
