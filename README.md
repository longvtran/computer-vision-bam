# computer-vision-bam

## How to run

### Get the dataset
To obtain the BAM dataset, follow the instructions [here](https://bam-dataset.org/). Once finished, you should have be able to download a `.sqlite` file. Save this file in the /data folder. The download script requires `sqlite3` and `parallel`, which can be obtained by running `sudo apt install sqlite3 parallel`. Then run the download script from the /data folder:
```
./get_data.sh
``` 
To process the data into the appropriate format, run:
```
python main.py --mode=gen_data --update
```

If the data is already saved, run:
```
python main.py --mode=gen_data --no-update
```

### Run training
To train your model, run:

```
python main.py --mode=train
```

