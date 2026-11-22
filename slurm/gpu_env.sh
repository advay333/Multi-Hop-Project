# Sourced (not run) by every GPU job script:  source slurm/gpu_env.sh
# Activates the venv, makes jobs offline, and makes sure Triton (the GPU-kernel compiler inside
# vLLM / torch.compile) can build its small C helper on the compute node. That helper needs
# Python's C headers (Python.h); the first reader_smoke (2026-11-22) failed with
# "gcc ... cuda_utils.c ... returned non-zero exit status 1" because the system has no
# python3.11-devel. setup_reader.sh puts a copy of the headers in ~/mhtable-pyinclude; gcc finds
# them through C_INCLUDE_PATH.
source "$HOME/mhtable-venv/bin/activate"
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"

PYINC="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [ ! -f "$PYINC/Python.h" ] && [ -f "$HOME/mhtable-pyinclude/Python.h" ]; then
  export C_INCLUDE_PATH="$HOME/mhtable-pyinclude${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi

# Same kind of build Triton does (include Python.h, link the GPU driver), so a failure shows
# gcc's own message here instead of deep inside vLLM's start-up.
_cc_dir="$(mktemp -d)"
printf '#include <Python.h>\n#include <stdio.h>\nint f(void){return 0;}\n' > "$_cc_dir/t.c"
if gcc "$_cc_dir/t.c" -shared -fPIC -o "$_cc_dir/t.so" -I"$PYINC" -l:libcuda.so.1 -L/lib64 2> "$_cc_dir/err"; then
  echo "== C build check (Python.h + libcuda): OK${C_INCLUDE_PATH:+ (headers from $C_INCLUDE_PATH)}"
else
  echo "== C build check FAILED - Triton will fail the same way. gcc said:"; cat "$_cc_dir/err"
fi
rm -rf "$_cc_dir"
