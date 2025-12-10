nuitka  --python-for-scons=/app/base/bin/python \
    --include-module=models \
    --include-module=datasets \
    --include-module=losses \
    --include-module=utils \
    --include-data-dir=pretrain=pretrain \
    --output-filename=rcmvsnet \
    --remove-output \
    eval_rcmvsnet_dtu.py

nuitka --python-for-scons=/app/base/bin/python --remove-output --output-filename=col2mvs colmap2mvsnet.py