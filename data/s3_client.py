import boto3
import botocore
import pandas as pd
import pyarrow.parquet as pq
import io
from concurrent.futures import ThreadPoolExecutor
import hashlib
from PIL import Image
import warnings

class s3_client:
    def __init__(self, 
                 access_key=None,
                 secret_key=None,
                 bucket_name='-----',
                 data_prefix='',
                 endpoint='http://10.140.14.204',
                 **args):
        self.bucket_name = bucket_name
        self.data_prefix = data_prefix

        # 检查 Access Key 和 Secret Key 是否提供
        assert access_key is not None and secret_key is not None, 'AK and SK must be specified in the dataset config file!!!'

        # 配置 boto3 客户端
        client_config = botocore.config.Config(
            max_pool_connections=2000,
        )
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=client_config
        )

    def list_files(self):
        """
        列出 S3 桶中指定前缀的数据文件
        """
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.data_prefix)
        if 'Contents' in response:
            files = [item['Key'] for item in response['Contents'] if item['Key'].endswith('.parquet')]
            return files
        else:
            return []

    def read_parquet(self, file_key):
        """
        从指定的 S3 桶文件中读取 Parquet 数据，并返回 Pandas DataFrame
        """
        try:
            # print(f"Reading Parquet file: {file_key}")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            # 从 S3 响应流中读取 Parquet 文件
            parquet_data = io.BytesIO(response['Body'].read())
            table = pq.read_table(parquet_data, columns=["URL", "top_caption"])
            df = table.to_pandas()
            return df
        except Exception as e:
            print(f"Error reading parquet file {file_key}: {e}")
            return None

    def read_all_parquet_files(self, max_workers=8, clips=[0, 10]):
        """
        并发读取多个 Parquet 文件并合并为一个 Pandas DataFrame
        """
        # 列出所有 Parquet 文件
        parquet_files = self.list_files()
        # print(f"Found {len(parquet_files)} Parquet files in the bucket.")

        if not parquet_files:
            print("No Parquet files found.")
            return None

        # 使用线程池并发读取文件
        all_dataframes = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.read_parquet, parquet_files[clips[0]:clips[1]]))

        # 收集所有非空 DataFrame
        all_dataframes = [df for df in results if df is not None]

        # 合并所有 DataFrame
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            # print(f"Successfully combined {len(all_dataframes)} DataFrames.")
            return combined_df
        else:
            print("Failed to read any Parquet files.")
            return None

    def download_image(self, image_hash):
        """
        根据哈希值从 S3 中下载对应的图片文件
        """
        # 构造图片的 S3 Key
        image_key = f"laion-coco/images/{image_hash}.jpg"
        print(f"Trying to download image: {image_key}")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=image_key)
            # 将图片保存到本地
            with open(f"{image_hash}.jpg", "wb") as f:
                f.write(response['Body'].read())
            print(f"Image {image_hash}.jpg downloaded successfully.")
        except Exception as e:
            print(f"Error downloading image {image_key}: {e}")

    def read_image_from_s3(self, url):
        image_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        url = f"laion-coco/images/{image_hash}.jpg"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=url)
            # 将图片保存到本地
            image_data = response['Body'].read()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            # print(f"Error loading image {url}: {e}")  
            pass



if __name__ == "__main__":
    s3_cfg=dict(access_key='FB7QKWTWP279SQMLBX4H',
                secret_key= 'dN6ph2f9cQcVhnOCngiGKwPUjMqpM9o4oiKM67mb',
                bucket_name='public-dataset',
                data_prefix = "laion-coco/meta/",
                endpoint='http://p-ceph-norm-outside.pjlab.org.cn')

    s3 = s3_client(**s3_cfg)

    # 并发读取并合并所有 Parquet 文件
    combined_df = s3.read_all_parquet_files(max_workers=16, clips=[0,1])

    if combined_df is not None:
        print("Combined DataFrame:")
        print(combined_df.head())
        print(f"Total rows: {len(combined_df)}")

    # 获取第一行 URL
    for i in range(10000):
        first_row = combined_df.iloc[i]
        url = first_row['URL']
        print(f"First row URL: {url}")
        image_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        image = s3.read_image_from_s3(url)
    # s3.download_image(image_hash)

    import pdb; pdb.set_trace()