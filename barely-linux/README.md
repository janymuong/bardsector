# barely linux
> busybox & co

a minimal linux distro: the linux kernel + busybox, packed into an initramfs and booted in qemu.

## build and run

```bash
chmod +x build.sh
./build.sh        # downloads sources, builds kernel + busybox, packs initramfs
./build.sh run    # boots it in qemu
```

## dependencies (arch, btw):

```bash
# use the equivalent fo your distro
sudo pacman -S --needed base-devel bc cpio wget xz openssl libelf qemu-system-x86
```


## how it boots

1. qemu loads `out/bzImage` (the kernel) and `out/initramfs.cpio.gz` (the root filesystem, held in ram).
2. the kernel unpacks the initramfs and runs `/init`.
3. `/init` mounts `/proc`, `/sys`, `/dev`, `/tmp`, prints the banner, then starts a busybox shell.
4. every command you type (`ls`, `cat`, `ps`...) is a symlink to the single `busybox` binary.

## NOTE: 
- edit `work/rootfs/init` or `etc/motd` to change the boot behavior and branding, then re-pack the initramfs.
- add a real init (busybox `init` with `/etc/inittab`) and a login prompt.
- add networking: `udhcpc` for dhcp, a qemu `-nic user` option.
- add a package: compile a static program and copy it into `rootfs/bin`.
- boot on real hardware: make a disk image with a bootloader (grub or syslinux) plus a persistent root filesystem.
