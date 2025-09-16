import logging
import os
import xmltodict
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_s3_uri(s3_uri: str = None):
    """Helper function to parse an S3 URI into bucket and key parts"""
    if s3_uri is None:
        raise ValueError("s3_uri cannot be None")

    if s3_uri.startswith('s3://'):
        s3_uri = s3_uri[5:]
    parts = s3_uri.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    if prefix and prefix.endswith('/'):
        prefix = prefix[:-1]
    return bucket, prefix


def download_task_from_s3(task_id: int, s3_dataset_cache: str = None) -> bool:
    import boto3
    """
    Downloads the task and dataset from S3 if available.
    """
    if s3_dataset_cache is None:
        return False

    # OpenML cache directory location for tasks + datasets
    local_cache_dir = Path(os.path.expanduser("~/.cache/openml/org/openml/www"))
    os.makedirs(local_cache_dir, exist_ok=True)

    try:
        s3_client = boto3.client('s3')
        s3_bucket, s3_prefix = parse_s3_uri(s3_uri=s3_dataset_cache)

        task_cache_dir = local_cache_dir / "tasks" / str(task_id)
        os.makedirs(task_cache_dir, exist_ok=True)

        logger.info(f"Attempting to download task {task_id} from S3 bucket {s3_bucket}")
        s3_key_prefix = f"{s3_prefix}/tasks/{task_id}/org/openml/www/tasks/{task_id}"

        try:
            task_xml_path = task_cache_dir / "task.xml"
            if not task_xml_path.exists():
                s3_client.download_file(
                    Bucket=s3_bucket,
                    Key=f"{s3_key_prefix}/task.xml",
                    Filename=str(task_cache_dir / "task.xml")
                )
                logger.info(f"Downloaded task.xml for task {task_id} from S3")
            else:
                logger.info(f"task.xml already exists for task {task_id}, skipping download")

            datasplits_path = task_cache_dir / "datasplits.arff"
            if not datasplits_path.exists():
                try:
                    s3_client.download_file(
                        Bucket=s3_bucket,
                        Key=f"{s3_key_prefix}/datasplits.arff",
                        Filename=str(task_cache_dir / "datasplits.arff")
                    )
                    logger.info(f"Downloaded datasplits.arff for task {task_id} from S3")
                except s3_client.exceptions.ClientError:
                    logger.info(f"No datasplits.arff found in S3 for task {task_id}")
            else:
                logger.info(f"datasplits.arff already exists for task {task_id}, skipping download")

            with open(task_cache_dir / "task.xml", 'r') as f:
                task_xml = f.read()

            task_dict = xmltodict.parse(task_xml)["oml:task"]
            dataset_id = None
            if isinstance(task_dict["oml:input"], list):
                for input_item in task_dict["oml:input"]:
                    if input_item["@name"] == "source_data":
                        dataset_id = int(input_item["oml:data_set"]["oml:data_set_id"])
                        break
            else:
                if task_dict["oml:input"]["@name"] == "source_data":
                    dataset_id = int(task_dict["oml:input"]["oml:data_set"]["oml:data_set_id"])

            if dataset_id is not None:
                dataset_id_str = str(dataset_id)
                dataset_cache_dir = local_cache_dir / "datasets" / dataset_id_str
                os.makedirs(dataset_cache_dir, exist_ok=True)

                logger.info(f"Attempting to download dataset {dataset_id} data from S3")
                s3_dataset_prefix = f"{s3_prefix}/tasks/{task_id}/org/openml/www/datasets/{dataset_id}"

                try:
                    response = s3_client.list_objects_v2(
                        Bucket=s3_bucket,
                        Prefix=f"{s3_dataset_prefix}/"
                    )

                    if 'Contents' in response:
                        for obj in response['Contents']:
                            s3_key = obj['Key']
                            filename = os.path.basename(s3_key)
                            if filename == '':
                                continue

                            local_file_path = dataset_cache_dir / filename
                            if not local_file_path.exists():
                                try:
                                    s3_client.download_file(
                                        Bucket=s3_bucket,
                                        Key=s3_key,
                                        Filename=str(local_file_path)
                                    )
                                    logger.info(f"Downloaded {filename} for dataset {dataset_id} from S3")
                                except s3_client.exceptions.ClientError as e:
                                    logger.info(f"Error downloading {filename} for dataset {dataset_id}: {e}")
                            else:
                                logger.info(f"{filename} already exists for dataset {dataset_id}, skipping download")
                    else:
                        logger.info(f"No files found in S3 for dataset {dataset_id}")
                except s3_client.exceptions.ClientError as e:
                    logger.warning(f"Error listing objects in S3 for dataset {dataset_id}: {e}")

            logger.info(f"Successfully downloaded task {task_id} from S3")
            return True

        except s3_client.exceptions.ClientError as e:
            logger.info(f"Task {task_id} not found in S3 ({e}), now trying OpenML server...")
            return False

    except Exception as e:
        logger.error(f"Error accessing S3 for task {task_id}: {str(e)}")
        return False
