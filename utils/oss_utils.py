from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
from qcloud_cos.cos_exception import CosClientError, CosServiceError

class TencentOss(): 
    def __init__(self): 
        SECRET_ID = 'AKIDpUv51h8JHyOYR3OTHp66pITN0NUKtWtz'
        SECRET_KEY = 'rSmqeCUYMt99sKVQ5VfwynU0LWUFqFUW'
        REGION = 'ap-guangzhou' 
        self.BUCKET = 'thingx-1326665451'
        config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
        self.client = CosS3Client(config)

    def upload_file(self, local_file, cos_key):
        try:
            response = self.client.upload_file(
                Bucket=self.BUCKET,
                Key=cos_key,          # COS上的文件路径+文件名
                LocalFilePath=local_file  # 本地文件路径
            )
            print(f"文件上传成功：{response['ETag']}")
            return True
        except Exception as e:
            print(f"上传失败：{str(e)}")
            return False 

    def download_file(self, cos_key, local_file):
        try:
            response = self.client.download_file(
                Bucket=self.BUCKET,
                Key=cos_key,          # COS上的文件路径+文件名
                DestFilePath=local_file  # 本地保存路径
            )
            print("文件下载成功")
            return True
        except Exception as e:
            print(f"下载失败：{str(e)}")
            return False

    def create_folder(self, folder_path):
        """
        在 COS 中创建文件夹
        :param folder_path: 文件夹路径(如 "documents/2023/")
        """
        try:
            # 关键：必须以 "/" 结尾，并上传一个 0 字节的空对象
            if not folder_path.endswith('/'):
                folder_path += '/'

            # 上传空对象（创建文件夹）
            response = self.client.put_object(
                Bucket=self.BUCKET,
                Key=folder_path,  # Key 必须以 / 结尾
                Body=b''          # 空内容
            )
            print(f"文件夹创建成功：{folder_path}")
            return True
        except CosClientError as e:
            print(f"客户端错误：{str(e)}")
            return False
        except CosServiceError as e:
            print(f"服务端错误：{e.get_error_code()}: {e.get_error_msg()}")
            return False
        
    def list_directory(self, target_path='', delimiter='/'):
        """
        列出指定路径下的所有文件和文件夹
        :param target_path: 目标路径（如 "documents/2023/"）
        :return: (文件列表, 文件夹列表)
        """
        target_path = target_path.lstrip('/')  # 标准化路径
        
        # 确保路径格式正确
        if target_path and not target_path.endswith('/'):
            target_path += '/'

        files = []
        folders = []
        marker = ''  # 分页标记
        is_truncated = True

        try:
            while is_truncated:
                response = self.client.list_objects(
                    Bucket=self.BUCKET,
                    Prefix=target_path,
                    Delimiter=delimiter,
                    Marker=marker
                )

                # 获取文件夹列表（CommonPrefixes）
                if 'CommonPrefixes' in response:
                    folders += [prefix['Prefix'] for prefix in response['CommonPrefixes']]

                # 获取文件列表（过滤掉文件夹标记）
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        # 排除文件夹标记对象（大小为0且以/结尾）
                        if not (obj['Size'] == 0 and key.endswith('/')):
                            files.append({
                                'Key': key,
                                'Size': obj['Size'],
                                'LastModified': obj['LastModified']
                            })

                # 检查是否还有更多内容
                is_truncated = response.get('IsTruncated', 'false').lower() == 'true'
                marker = response.get('NextMarker', '') if is_truncated else ''

            return files, folders

        except CosServiceError as e:
            print(f"请求失败：{e.get_error_code()} - {e.get_error_msg()}")
            return [], []

    def get_oss_url(self, key, expired=3600):
        url = self.client.get_presigned_download_url(
            Bucket=self.BUCKET,
            Key=key,
            Expired=expired  # 有效期（秒），最大 7 天（604800 秒）
        )
        return url
    
    def delete_file(self, cos_key):
        try:
            response = self.client.delete_object(
                Bucket=self.BUCKET,
                Key=cos_key  # COS 上的文件路径+文件名
            )
            print(f"文件删除成功：{cos_key}")
            return True
        except CosClientError as e:
            print(f"客户端错误：{str(e)}")
            return False
        except CosServiceError as e:
            print(f"服务端错误：{e.get_error_code()}: {e.get_error_msg()}")
            return False
    
    def delete_folder(self, folder_path):
        try:
            # 确保路径格式正确，必须以 "/" 结尾
            if not folder_path.endswith('/'):
                folder_path += '/'

            # 列出文件夹下的所有文件和子文件夹
            files, folders = self.list_directory(folder_path)

            # 删除所有文件
            for file in files:
                self.delete_file(file['Key'])

            # 递归删除子文件夹
            for folder in folders:
                self.delete_folder(folder)

            print(f"文件夹删除成功：{folder_path}")
            return True

        except CosClientError as e:
            print(f"客户端错误：{str(e)}")
            return False
        except CosServiceError as e:
            print(f"服务端错误：{e.get_error_code()}: {e.get_error_msg()}")
            return False
    
    def print_list(self, path):
        data = self.list_directory(path)
        # 解构数据
        dict_list, additional_info = data
        # 打印字典列表
        for item in dict_list:
            print("Key:", item['Key'])
            print("Size:", item['Size'])
            print("Last Modified:", item['LastModified'])
            print("-" * 40)  # 分隔线，使输出更易读

        # 如果需要处理第二个列表的信息，可以在这里添加相应的逻辑
        print("Additional Info:")
        for info in additional_info:
            print(info)    

    def upload_folder(self, local_folder_path, cos_folder_path=''):
        """
        上传本地文件夹到 COS
        :param local_folder_path: 本地文件夹路径（如 "/path/to/local/folder"）
        :param cos_folder_path: COS 上的目标文件夹路径（如 "documents/2023/"）
        """
        try:
            # 确保本地文件夹路径存在
            if not os.path.isdir(local_folder_path):
                print(f"本地文件夹不存在：{local_folder_path}")
                return False

            # 确保 COS 文件夹路径以 "/" 结尾
            if cos_folder_path and not cos_folder_path.endswith('/'):
                cos_folder_path += '/'

            # 遍历本地文件夹中的所有文件和子文件夹
            for root, dirs, files in os.walk(local_folder_path):
                for file_name in files:
                    # 构造本地文件路径
                    local_file_path = os.path.join(root, file_name)

                    # 计算相对路径
                    relative_path = os.path.relpath(local_file_path, local_folder_path)

                    # 构造 COS 上的目标路径
                    cos_key = os.path.join(cos_folder_path, relative_path).replace("\\", "/")

                    # 上传文件
                    self.upload_file(local_file_path, cos_key)

            print(f"文件夹上传成功：{local_folder_path} -> {cos_folder_path}")
            return True

        except Exception as e:
            print(f"文件夹上传失败：{str(e)}")
            return False
        
    def download_folder(self, cos_folder_path, local_folder_path):
        """
        从 COS 下载整个文件夹到本地
        :param cos_folder_path: COS 上的文件夹路径（如 "documents/2023/"）
        :param local_folder_path: 本地保存的目标文件夹路径（如 "/path/to/local/folder"）
        :return: 成功与否
        """
        try:
            # 确保 COS 文件夹路径以 '/' 结尾
            if not cos_folder_path.endswith('/'):
                cos_folder_path += '/'

            # 创建本地文件夹（如果不存在）
            os.makedirs(local_folder_path, exist_ok=True)

            # 列出该路径下的所有文件和子文件夹
            files, folders = self.list_directory(cos_folder_path)

            if not files and not folders:
                print(f"COS 文件夹为空或不存在：{cos_folder_path}")
                return False

            # 下载所有文件
            for file_info in files:
                cos_key = file_info['Key']

                # 构造本地文件路径
                relative_path = os.path.relpath(cos_key, cos_folder_path)
                local_file_path = os.path.join(local_folder_path, relative_path)

                # 创建父目录（防止路径不存在）
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # 调用下载方法
                print(f"正在下载：{cos_key} -> {local_file_path}")
                self.download_file(cos_key, local_file_path)

            print(f"文件夹下载完成：{cos_folder_path} -> {local_folder_path}")
            return True

        except Exception as e:
            print(f"下载文件夹失败：{str(e)}")
            return False