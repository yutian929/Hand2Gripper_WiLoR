"""
Hand2Gripper WiLoR Module

This module provides a unified interface for hand detection and 3D reconstruction
using the WiLoR model. It encapsulates the core functionality from the original
WiLoR implementation into easy-to-use classes.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch.utils.data
from ultralytics import YOLO
try:
    from .wilor.models import WiLoR, load_wilor
    from .wilor.utils import recursive_to
    from .wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    from .wilor.utils.renderer import Renderer, cam_crop_to_full
except ImportError as e:
    print(f"Warning: Could not import WiLoR modules: {e}")
    print("Make sure you're running from the correct directory with WiLoR installed")

# Constants
LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


class HandDetector:
    """YOLO-based hand detection module"""
    
    def __init__(self, model_path: str=os.path.join(os.path.dirname(__file__), 'pretrained_models', 'detector.pt'), device: str = 'auto'):
        """
        Initialize hand detector
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = device
        self.model = YOLO(model_path)
        if device != 'auto':
            self.model = self.model.to(device)
    
    def detect_hands(self, image: np.ndarray, conf_threshold: float = 0.3, 
                    iou_threshold: float = 0.5) -> Tuple[List[List[float]], List[bool]]:
        """
        Detect hands in image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            
        Returns:
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            is_right: List of boolean values indicating right hand (True) or left hand (False)
        """
        detections = self.model(image, conf=conf_threshold, verbose=False, iou=iou_threshold)[0]
        
        bboxes = []
        is_right = []
        
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            hand_side = det.boxes.cls.cpu().detach().squeeze().item()
            bboxes.append(bbox[:4].tolist())
            is_right.append(bool(hand_side))  # 0=left, 1=right
        
        return bboxes, is_right


class WiLoRModel:
    """WiLoR 3D hand reconstruction model"""
    
    def __init__(self, model_path: str=os.path.join(os.path.dirname(__file__), 'pretrained_models', 'wilor_final.ckpt'), config_path: str=os.path.join(os.path.dirname(__file__), 'pretrained_models', 'model_config.yaml'), device: str = 'auto'):
        """
        Initialize WiLoR model
        
        Args:
            model_path: Path to WiLoR model checkpoint
            config_path: Path to model config file
            device: Device to run inference on
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() and device != 'cpu' else torch.device('cpu')
        
        # Load WiLoR model
        self.model, self.model_cfg = load_wilor(
            checkpoint_path=model_path,
            cfg_path=config_path
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def reconstruct_hands(self, image: np.ndarray, bboxes: List[List[float]], 
                         is_right: List[bool], rescale_factor: float = 2.0) -> Dict:
        """
        Reconstruct 3D hands from detected bounding boxes
        
        Args:
            image: Input image (BGR format)
            bboxes: List of bounding boxes
            is_right: List of hand sides
            rescale_factor: Factor for padding the bbox
            
        Returns:
            Dictionary containing reconstruction results
        """
        if len(bboxes) == 0:
            return {
                'vertices': [],
                'joints': [],
                'cam_t': [],
                'is_right': [],
                'kpts_2d': []
            }
        
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        
        # Create dataset and dataloader
        dataset = ViTDetDataset(self.model_cfg, image, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints = []
        all_kpts = []
        
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            
            with torch.no_grad():
                out = self.model(batch)
            
            # Process camera parameters
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            # Process each hand in the batch
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right_hand = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
                joints[:, 0] = (2 * is_right_hand - 1) * joints[:, 0]
                
                cam_t = pred_cam_t_full[n]
                kpts_2d = self._project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_hand)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
        
        return {
            'vertices': all_verts,
            'joints': all_joints,
            'cam_t': all_cam_t,
            'is_right': all_right,
            'kpts_2d': all_kpts,
            'focal_length': scaled_focal_length,
            'img_size': img_size[n] if len(all_verts) > 0 else None
        }
    
    def _project_full_img(self, points: np.ndarray, cam_trans: np.ndarray, 
                         focal_length: float, img_res: torch.Tensor) -> np.ndarray:
        """Project 3D points to 2D image coordinates"""
        camera_center = [img_res[0] / 2., img_res[1] / 2.]
        K = torch.eye(3)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[0, 2] = camera_center[0]
        K[1, 2] = camera_center[1]
        
        points = points + cam_trans
        points = points / points[..., -1:]
        
        V_2d = (K @ points.T).T
        return V_2d[..., :-1]


class HandRenderer:
    """3D hand mesh renderer for visualization"""
    
    def __init__(self, model_cfg=os.path.join(os.path.dirname(__file__), 'pretrained_models', 'model_config.yaml'), faces: np.ndarray=np.array([[0, 1, 2], [1, 2, 3]])):
        breakpoint()
        self.model_cfg = model_cfg
        self.faces = faces
        self.renderer = Renderer(model_cfg, faces=faces)
    
    def render_hands(self, vertices_list: List[np.ndarray], cam_t_list: List[np.ndarray],
                    is_right_list: List[bool], img_size: np.ndarray, 
                    focal_length: float, mesh_color: Tuple[float, float, float] = LIGHT_PURPLE) -> np.ndarray:
        """
        Render multiple 3D hands
        
        Args:
            vertices_list: List of hand vertices
            cam_t_list: List of camera translations
            is_right_list: List of hand sides
            img_size: Image size
            focal_length: Camera focal length
            mesh_color: RGB color for mesh
            
        Returns:
            Rendered image with alpha channel
        """
        if len(vertices_list) == 0:
            return np.zeros((int(img_size[1]), int(img_size[0]), 4), dtype=np.float32)
        
        misc_args = dict(
            mesh_base_color=mesh_color,
            scene_bg_color=(1, 1, 1),
            focal_length=focal_length,
        )
        
        cam_view = self.renderer.render_rgba_multiple(
            vertices_list, 
            cam_t=cam_t_list, 
            render_res=img_size, 
            is_right=is_right_list, 
            **misc_args
        )
        
        return cam_view
    
    def vertices_to_trimesh(self, vertices: np.ndarray, camera_translation: np.ndarray,
                          color: Tuple[float, float, float] = LIGHT_PURPLE,
                          is_right: bool = True):
        """
        Convert vertices to trimesh object
        
        Args:
            vertices: Hand vertices (N, 3)
            camera_translation: Camera translation (3,)
            color: RGB color for mesh
            is_right: Whether this is a right hand
            
        Returns:
            Trimesh object
        """
        return self.renderer.vertices_to_trimesh(vertices, camera_translation, color, is_right=is_right)


class Hand2GripperWiLoR:
    """Main Hand2Gripper WiLoR pipeline"""
    
    def __init__(self, model_dir: str = './pretrained_models/', mano_data_dir: str = './mano_data/', 
                 device: str = 'auto'):
        """
        Initialize the complete Hand2Gripper WiLoR pipeline
        
        Args:
            model_dir: Directory containing model files
            mano_data_dir: Directory containing MANO data
            device: Device to run inference on
        """
        self.device = device
        self.model_dir = model_dir
        self.mano_data_dir = mano_data_dir
        
        # Initialize components
        self.detector = HandDetector(
            model_path=os.path.join(model_dir, 'detector.pt'),
            device=device
        )
        
        self.wilor_model = WiLoRModel(
            model_path=os.path.join(model_dir, 'wilor_final.ckpt'),
            config_path=os.path.join(model_dir, 'model_config.yaml'),
            device=device
        )
        
        self.renderer = HandRenderer(
            model_cfg=self.wilor_model.model_cfg,
            faces=self.wilor_model.model.mano.faces
        )
    
    def process_image(self, image: np.ndarray, conf_threshold: float = 0.3,
                     rescale_factor: float = 2.0, save_mesh: bool = False,
                     output_dir: str = './output') -> Dict:
        """
        Process single image for hand detection and reconstruction
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Detection confidence threshold
            rescale_factor: Factor for padding the bbox
            save_mesh: Whether to save 3D meshes
            output_dir: Output directory for saved files
            
        Returns:
            Dictionary containing detection and reconstruction results
        """
        # Detect hands
        bboxes, is_right = self.detector.detect_hands(image, conf_threshold)
        
        if len(bboxes) == 0:
            return {
                'image': image,
                'detections': [],
                'reconstructions': [],
                'rendered_image': image
            }
        
        # Reconstruct 3D hands
        reconstruction_results = self.wilor_model.reconstruct_hands(
            image, bboxes, is_right, rescale_factor
        )
        
        # Save meshes if requested
        if save_mesh and len(reconstruction_results['vertices']) > 0:
            os.makedirs(output_dir, exist_ok=True)
            for i, (verts, cam_t, is_right_hand) in enumerate(zip(
                reconstruction_results['vertices'],
                reconstruction_results['cam_t'],
                reconstruction_results['is_right']
            )):
                mesh = self.renderer.vertices_to_trimesh(verts, cam_t, is_right=is_right_hand)
                mesh.export(os.path.join(output_dir, f'hand_{i}.obj'))
        
        # Render results
        if len(reconstruction_results['vertices']) > 0:
            cam_view = self.renderer.render_hands(
                reconstruction_results['vertices'],
                reconstruction_results['cam_t'],
                reconstruction_results['is_right'],
                reconstruction_results['img_size'],
                reconstruction_results['focal_length']
            )
            
            # Overlay on original image
            input_img = image.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            rendered_image = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)
        else:
            rendered_image = image
        
        # Prepare results
        detections = []
        for i, (bbox, is_right_hand) in enumerate(zip(bboxes, is_right)):
            detection = {
                'bbox': bbox,
                'is_right': is_right_hand,
                'vertices': reconstruction_results['vertices'][i] if i < len(reconstruction_results['vertices']) else None,
                'joints': reconstruction_results['joints'][i] if i < len(reconstruction_results['joints']) else None,
                'camera_params': reconstruction_results['cam_t'][i] if i < len(reconstruction_results['cam_t']) else None
            }
            detections.append(detection)
        
        return {
            'image': image,
            'detections': detections,
            'reconstructions': reconstruction_results,
            'rendered_image': rendered_image
        }
    
    def process_folder(self, img_folder: str, out_folder: str, 
                      file_types: List[str] = None, **kwargs) -> List[Dict]:
        """
        Process all images in a folder
        
        Args:
            img_folder: Input folder path
            out_folder: Output folder path
            file_types: List of file extensions to process
            **kwargs: Additional arguments for process_image
            
        Returns:
            List of result dictionaries
        """
        if file_types is None:
            file_types = ['*.jpg', '*.png', '*.jpeg']
        
        os.makedirs(out_folder, exist_ok=True)
        
        # Get all image paths
        img_paths = [img for end in file_types for img in Path(img_folder).glob(end)]
        
        results = []
        for img_path in img_paths:
            print(f"Processing: {img_path}")
            
            # Load image
            img_cv2 = cv2.imread(str(img_path))
            if img_cv2 is None:
                print(f"Could not load image: {img_path}")
                continue
            
            # Process image
            result = self.process_image(img_cv2, **kwargs)
            
            # Save rendered result
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            output_path = os.path.join(out_folder, f'{img_fn}.jpg')
            cv2.imwrite(output_path, result['rendered_image'])
            
            # Save meshes if requested
            if kwargs.get('save_mesh', False) and len(result['detections']) > 0:
                for i, detection in enumerate(result['detections']):
                    if detection['vertices'] is not None:
                        mesh = self.renderer.vertices_to_trimesh(
                            detection['vertices'], 
                            detection['camera_params'],
                            is_right=detection['is_right']
                        )
                        mesh_path = os.path.join(out_folder, f'{img_fn}_{i}.obj')
                        mesh.export(mesh_path)
            
            results.append(result)
            print(f"  Detected {len(result['detections'])} hands")
        
        return results
