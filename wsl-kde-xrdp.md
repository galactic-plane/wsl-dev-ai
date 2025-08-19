
# WSL2 Ubuntu 24.04 with KDE Plasma Desktop & XRDP Setup Guide

## Table of Contents
- [WSL2 Ubuntu 24.04 with KDE Plasma Desktop \& XRDP Setup Guide](#wsl2-ubuntu-2404-with-kde-plasma-desktop--xrdp-setup-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Install WSL2 and a Linux Distribution](#1-install-wsl2-and-a-linux-distribution)
  - [2. Update and Upgrade Ubuntu](#2-update-and-upgrade-ubuntu)
  - [3. Install Tasksel and Desktop Environment (KDE Plasma)](#3-install-tasksel-and-desktop-environment-kde-plasma)
  - [4. Install and Configure XRDP](#4-install-and-configure-xrdp)
  - [5. Find Your Linux IP Address](#5-find-your-linux-ip-address)
  - [6. Connect with Windows Remote Desktop](#6-connect-with-windows-remote-desktop)


## 1. Install WSL2 and a Linux Distribution

**Run PowerShell as Administrator:**

```powershell
# Check if WSL is installed
wsl --list --verbose
# If not installed
wsl --install
```

Restart your system if prompted.

**List available distributions:**

```powershell
wsl --list --online
```

**Install Ubuntu 24.04:**

```powershell
wsl --install -d Ubuntu-24.04
```

After installation, set up your **username** and **password** inside Ubuntu.

**Verify distribution:**

```bash
lsb_release -a
```

---

## 2. Update and Upgrade Ubuntu

**Inside Ubuntu shell:**

```bash
sudo apt update
sudo apt upgrade -y
```

---


## 3. Install Tasksel and Desktop Environment (KDE Plasma)

```bash
sudo apt install tasksel -y
sudo tasksel
```

In the menu:
- Select **KDE Plasma** (or KDE if listed) with <kbd>Space</kbd>
- Hit <kbd>Tab</kbd> → <kbd>OK</kbd> → <kbd>Enter</kbd>

Wait for installation (several minutes).

---

## 4. Install and Configure XRDP

```bash
sudo apt install xrdp -y
sudo systemctl status xrdp
```

Check that the status shows **active (running)**. Exit status with <kbd>Ctrl</kbd>+<kbd>C</kbd>.

---

## 5. Find Your Linux IP Address

```bash
ip addr
```

Copy the value next to `inet` for your adapter (e.g., `172.20.224.1`).

---

## 6. Connect with Windows Remote Desktop

From Windows:

1. Open **Remote Desktop Connection** (`mstsc`).
2. Enter the IP address from step 5.
3. At the login screen:
    - **Session type:** choose `Xorg`
    - **Username:** your Ubuntu username
    - **Password:** your Ubuntu password

Once connected, you’ll land in the KDE Plasma desktop environment.
