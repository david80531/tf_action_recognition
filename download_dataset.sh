cd ..

if [ -d $PWD"/action_dataset" ]; then
    # 目錄 /path/to/dir 存在
    echo "action_dataset already exist."
else
    wget "http://crcv.ucf.edu/data/UCF101/UCF101.rar"

    unrar x UCF101.rar

    rm -rf UCF101.rar

    mv ./UCF-101 ./action_dataset
    echo "action_dataset install finished"
fi
