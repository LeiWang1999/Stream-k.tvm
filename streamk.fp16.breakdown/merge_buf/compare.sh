cd /root/BitBLAS/3rdparty/tvm/build

make -j

cd /root/Stream-k.tvm/streamk.fp16.breakdown/merge_buf

python /root/Stream-k.tvm/streamk.fp16.breakdown/merge_buf/streamk_fp16_ladder_block_reduce_q4_first_waves.py 2>&1 | tee first_waves.log

python /root/Stream-k.tvm/streamk.fp16.breakdown/merge_buf/streamk_fp16_ladder_block_reduce_q4_full_tiles.py 2>&1 | tee full_tiles.log
