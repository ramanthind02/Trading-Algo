# Remote SSH: Connect from Another Desktop

This machine (the **dev server**) runs the repo at `/home/raman/repos/Trading-Algo`. You can open this folder in Cursor on your **other desktop** via Remote SSH.

---

## Part A: On this machine (dev server) — one-time setup

Run these on the machine that has the repo (current machine, IP **192.168.1.71**, user **raman**).

### 1. Install and enable OpenSSH server

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
sudo systemctl status ssh   # should show "active (running)"
```

### 2. Allow your other desktop to log in with a key

On **this machine**, add the other desktop’s **public** key to `~/.ssh/authorized_keys`:

```bash
# Create file if missing and set permissions
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

Then **append** the public key from your other desktop (one line, no line break). Either:

- **Option A:** On the other desktop, run `cat ~/.ssh/id_ed25519.pub` (or `id_rsa.pub`) and copy the line. On this machine run:
  ```bash
  echo 'PASTE_THE_FULL_LINE_HERE' >> ~/.ssh/authorized_keys
  ```
- **Option B:** If you can SSH from the other desktop to this one with a password, from the **other desktop** run:
  ```bash
  ssh-copy-id raman@192.168.1.71
  ```
  (Enter your password once; after that, key login will work.)

### 3. (Optional) Find this machine’s IP if it changes

```bash
hostname -I | awk '{print $1}'
```

Use this IP in Part B as `HostName`. If your IP is static (e.g. 192.168.1.71), you can use that.

---

## Part B: On your other desktop — connect with Cursor

### 1. Install Cursor Remote SSH

- Open Cursor → Extensions (`Ctrl+Shift+X`)
- Search **Remote - SSH** → Install **Remote - SSH** (Cursor/Microsoft)

### 2. SSH key (if you don’t have one)

On the **other desktop** in a terminal:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519 -N ""
```

Then add the **public** key to this dev server (Part A, step 2).

### 3. SSH config on the other desktop

Edit or create **`~/.ssh/config`** on the **other desktop** (Windows: `C:\Users\YourName\.ssh\config`):

```
Host trading-dev
    HostName 192.168.1.71
    User raman
    IdentityFile ~/.ssh/id_ed25519
```

- Use **192.168.1.71** unless this dev server’s IP changed (see Part A step 3).
- The key path above matches the `ed25519` key generated in step 2; if you used a different key type or path, set `IdentityFile` accordingly (e.g. `~/.ssh/id_rsa` for an older RSA key).
- Save and run: `chmod 600 ~/.ssh/config` (Linux/macOS).

### 4. Connect in Cursor

- **Command Palette:** `Ctrl+Shift+P` → **Remote-SSH: Connect to Host…**
- Choose **trading-dev**
- When the remote window opens: **File → Open Folder** → enter:
  ```
  /home/raman/repos/Trading-Algo
  ```

### 5. Open from terminal (optional)

From the **other desktop** terminal:

```bash
cursor --folder-uri "vscode-remote://ssh-remote+trading-dev/home/raman/repos/Trading-Algo"
```

---

## Quick reference

| Role        | Machine        | IP (example)   | User  |
|------------|----------------|----------------|-------|
| Dev server | This machine   | 192.168.1.71   | raman |
| Client     | Other desktop  | —              | you   |

- **Venv on dev server:** `source /home/raman/repos/Trading-Algo/venv/bin/activate`
- **Security:** Only use this over a trusted network (or VPN). Prefer key-based login; avoid password auth in production.

> _Verified against current code via CodeGraph on 2026-06-07._
