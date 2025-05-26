# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

from getpass import getpass
from typing import Optional
from pathlib import Path
import netrc
import stat
import os


NETRC_MACHINE = "obz-backend"
NETRC_LOGIN = "user"
NETRC_PATH = Path.home() / ".netrc"


def read_netrc_key(machine: str = NETRC_MACHINE) -> Optional[str]:
    try:
        auth = netrc.netrc(NETRC_PATH).authenticators(machine)
        return auth[2] if auth else None
    except FileNotFoundError:
        return None
    except netrc.NetrcParseError as e:
        print(f"Error parsing .netrc: {e}")
        return None


def write_netrc_key(api_key: str, machine: str = NETRC_MACHINE, login: str = NETRC_LOGIN):
    entry = f"machine {machine} login {login} password {api_key}\n"
    netrc_file = NETRC_PATH

    if not netrc_file.exists():
        with netrc_file.open("w") as f:
            f.write(entry)
        os.chmod(netrc_file, stat.S_IRUSR | stat.S_IWUSR)
        print(f"Created new .netrc file with safe permissions at {netrc_file}")
        return

    # Ensure permissions are strict even on existing file
    current_mode = stat.S_IMODE(netrc_file.stat().st_mode)
    if current_mode != (stat.S_IRUSR | stat.S_IWUSR):
        os.chmod(netrc_file, stat.S_IRUSR | stat.S_IWUSR)
        print(f"Updated permissions for {netrc_file} to 600")

    try:
        existing_auth = netrc.netrc(netrc_file).authenticators(machine)
        if existing_auth:
            print(f"Entry for '{machine}' already exists in .netrc. Skipping.")
            return
    except netrc.NetrcParseError as e:
        print(f"Error parsing .netrc: {e}. Proceeding with append.")

    # Append new entry
    with netrc_file.open("a") as f:
        f.write("\n" + entry)
    print(f"Appended credentials for '{machine}' to existing .netrc.")


def get_api_key_interactively():
    print("Please log in to your OBZ account.")
    return getpass("Paste your API key: ").strip()