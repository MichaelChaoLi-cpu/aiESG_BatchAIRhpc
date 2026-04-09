# build the tf with conda
```
conda create --name tf python=3.9 -y
```
this is create the env
```
conda deactivate
conda activate tf

conda install -c conda-forge cudatoolkit=11.8.0 -y
```

```
pip install nvidia-cudnn-cu11==8.6.0.163
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```


```
pip install tensorflow==2.12.*
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

```
pip install google-cloud-bigquery==3.19.0 google-cloud-storage==2.16.0 joblib==1.3.2 \\
numpy==1.24.3 pandas==2.2.1 pandas-gbq==0.22.0 transformers==4.39.1 \\
db-dtypes==1.2.0 ipython==8.18.1 openpyxl==3.1.2 scikit-learn==1.4.0 \\
jupyter==1.0.0 matplotlib==3.8.3 notebook==7.1.2 xgboost==2.0.3
```

And add any other necessary package for this pipe