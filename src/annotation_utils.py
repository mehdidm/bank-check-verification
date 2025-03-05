import os
import json
import logging
from datetime import datetime
import label_studio_sdk


class LabelStudioAnnotator:
    """
    Manages annotation projects for bank check processing
    Handles project creation, data import, and annotation export
    """

    def __init__(self, host='http://localhost:8080', api_key=None):
        """
        Initialize Label Studio client

        Args:
            host (str): Label Studio server URL
            api_key (str, optional): API key for authentication
        """
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.api_key = api_key

        try:
            # Attempt to connect to Label Studio
            self.ls = label_studio_sdk.Client(
                url=self.host,
                api_key=self.api_key
            )
            self.logger.info("Label Studio client initialized successfully")
        except Exception as e:
            self.logger.error(f"Label Studio connection error: {e}")
            self.ls = None

    def create_project(self, filename, image_path):
        """
        Create a Label Studio annotation project for a specific check

        Args:
            filename (str): Original filename
            image_path (str): Path to preprocessed image

        Returns:
            label_studio_sdk.Project: Created project or None
        """
        if not self.ls:
            self.logger.warning("Label Studio client not initialized")
            return None

        try:
            # Load configuration
            config_path = 'label_studio_config/config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Generate unique project title
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_title = f"Check Annotation: {filename} - {timestamp}"

            # Create project
            project = self.ls.create_project(
                title=project_title,
                label_config=json.dumps(config['label_config'])
            )

            # Import image task
            project.import_tasks([{
                'data': {
                    'image': image_path
                }
            }])

            self.logger.info(f"Created annotation project for {filename}")
            return project

        except Exception as e:
            self.logger.error(f"Project creation error: {e}")
            return None

    def export_annotations(self, project_id, export_format='JSON'):
        """
        Export annotations from a project

        Args:
            project_id (int): ID of the project to export
            export_format (str): Format of exported annotations

        Returns:
            str: Path to exported annotation file
        """
        if not self.ls:
            self.logger.warning("Label Studio client not initialized")
            return None

        try:
            # Create export directory
            export_dir = 'data/annotated'
            os.makedirs(export_dir, exist_ok=True)

            # Generate export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exported_file = os.path.join(
                export_dir,
                f'annotations_project_{project_id}_{timestamp}.{export_format.lower()}'
            )

            # Export project annotations
            self.ls.export_project(
                project_id,
                exported_file,
                format=export_format
            )

            self.logger.info(f"Exported annotations to {exported_file}")
            return exported_file

        except Exception as e:
            self.logger.error(f"Annotation export error: {e}")
            return None