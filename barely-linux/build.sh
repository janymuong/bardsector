#!/usr/bin/env bash
# Minimal Linux distro: Linux kernel + BusyBox + initramfs, booted in QEMU.
# Usage: ./build.sh          (build everything)
#        ./build.sh run      (boot the result in QEMU)
set -euo pipefail

KERNEL_VER="6.12.10"
BUSYBOX_VER="1.36.1"
DISTRO_NAME="Barely Linux"
JOBS="$(nproc)"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$ROOT/work"
OUT="$ROOT/out"

run_qemu() {
  exec qemu-system-x86_64 \
    -kernel "$OUT/bzImage" \
    -initrd "$OUT/initramfs.cpio.gz" \
    -append "console=ttyS0 quiet" \
    -nographic -m 256M
}

if [ "${1:-}" = "run" ]; then
  run_qemu
fi

# --- 1. check build dependencies --------------------------------------------
missing=()
for tool in gcc make bc flex bison cpio gzip wget tar xz qemu-system-x86_64; do
  command -v "$tool" >/dev/null || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing tools: ${missing[*]}"
  echo "Arch:          sudo pacman -S --needed base-devel bc cpio wget xz openssl libelf qemu-system-x86"
  # echo "debian/ubuntu: sudo apt install build-essential bc flex bison libssl-dev libelf-dev cpio wget xz-utils qemu-system-x86"
  exit 1
fi

mkdir -p "$WORK" "$OUT"
cd "$WORK"

# --- 2. sources ----------------------------------------------------
[ -f "linux-$KERNEL_VER.tar.xz" ] || \
  wget "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-$KERNEL_VER.tar.xz"
[ -f "busybox-$BUSYBOX_VER.tar.bz2" ] || \
  wget "https://busybox.net/downloads/busybox-$BUSYBOX_VER.tar.bz2"

[ -d "linux-$KERNEL_VER" ] || tar xf "linux-$KERNEL_VER.tar.xz"
[ -d "busybox-$BUSYBOX_VER" ] || tar xf "busybox-$BUSYBOX_VER.tar.bz2"

# --- 3. build the kernel ----------------------------------------------------
if [ ! -f "$OUT/bzImage" ]; then
  echo ">> Building kernel (this takes a while)..."
  cd "$WORK/linux-$KERNEL_VER"
  make defconfig
  make -j"$JOBS" bzImage
  cp arch/x86/boot/bzImage "$OUT/bzImage"
  cd "$WORK"
fi

# --- 4. build a static busybox ----------------------------------------------
echo ">> Building BusyBox..."
cd "$WORK/busybox-$BUSYBOX_VER"
make defconfig
sed -i 's/^# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config
# 'tc' fails to compile against newer kernel headers and we don't need it
sed -i 's/^CONFIG_TC=y/# CONFIG_TC is not set/' .config
make -j"$JOBS"
make CONFIG_PREFIX="$WORK/rootfs" install
cd "$WORK"

# --- 5. assemble the root filesystem ----------------------------------------
echo ">> Assembling rootfs..."
cd "$WORK/rootfs"
mkdir -p proc sys dev tmp etc root

cat > etc/motd <<EOF

  Welcome to $DISTRO_NAME
  Linux $KERNEL_VER + BusyBox $BUSYBOX_VER
  Type 'poweroff' to shut down.

EOF

cat > init <<'EOF'
#!/bin/sh
mount -t proc     none /proc
mount -t sysfs    none /sys
mount -t devtmpfs none /dev
mount -t tmpfs    none /tmp
hostname barely-linux
cat /etc/motd
# cttyhack gives the shell a controlling terminal (job control, Ctrl-C)
exec setsid cttyhack /bin/sh
EOF
chmod +x init

# --- 6. pack the initramfs --------------------------------------------------
find . | cpio -o -H newc 2>/dev/null | gzip > "$OUT/initramfs.cpio.gz"

echo
echo "Done. Kernel and initramfs are in: $OUT"
echo "Boot it with: ./build.sh run"
