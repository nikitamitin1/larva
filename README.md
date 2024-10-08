# Larva v1.0 📷 🔑
Small and fast tool to face unlock your PAM on Linux

**DISLAIMER!** 🔴

*NO GUARANTEES are provided with this software!

NO GUARANTEE of workability!

This auth method utilita IS NOT SECURE!

It CAN NOT REPLACE other more secure ways to auth! 

USE AT YOUR OWN RISK!*


## Installation Guide 💾

Easy installation is described step-by-step. This service utilizes face recognition for authentication and integrates with PAM (Pluggable Authentication Module) on Linux systems.

## Prerequisites 🛍️

- **Python 3** and **pip** must be installed on your system.
- You need **sudo** privileges to perform the installation.

## Installation Steps ⚙️

Download **install.sh**


- Run

```bash
sudo <path_to_install.sh>/install.sh
```
and read dialog in shell

*If everything is ok after next login session (locked screen) 
1) your camera will try to recognize your face,
2) match the result with registered user faces in data storage and
3) unlock the screen if you are have access rights.

If it doesnt work try to check if **pam_exec.so** has been cofigured properly:

```bash
sudo nano /etc/pam.d/gdm-password
```

you should see line like this in the most first row:

```
auth        sufficient    pam_exec.so quiet /usr/bin/python3 /usr/local/larva/main_prod.py
```

if not you may add it manually and then push ```CTRL+X``` to save changes.


**!WARNING** 🔴

*Don't change anything but adding this particular string into the first row*

*Any changes may cause problem with PAM GUI authentication*

Enjoy! 
