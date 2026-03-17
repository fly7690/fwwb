"""
无人机航拍数据集准备脚本
功能：将VisDrone/UAVDT转换为YOLO格式，生成索引文件，支持场景分桶和数据增强
作者：B角色 - 数据与场景负责人
版本：V2.0-scene-bucket-with-aug
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
import yaml
from tqdm import tqdm
import glob
import shutil
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import albumentations as A


class SceneAnalyzer:
    """场景分析器：分析图像的场景特征"""

    # [保持原有的SceneAnalyzer类不变]
    def __init__(self):
        self.scene_tags = {
            'illumination': ['day', 'night', 'strong_light'],
            'weather': ['clear', 'foggy', 'rainy'],
            'density': ['sparse', 'medium', 'dense', 'very_dense'],
            'scale': ['mostly_small', 'mixed', 'mostly_large'],
            'altitude': ['low', 'medium', 'high'],
            'occlusion': ['none', 'little', 'heavy']
        }

    def analyze_illumination(self, img):
        """分析光照条件"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        if mean_brightness < 70:
            illumination = 'night'
        elif mean_brightness > 180:
            illumination = 'strong_light'
        else:
            illumination = 'day'

        return illumination, {'brightness': float(mean_brightness), 'contrast': float(std_brightness)}

    def analyze_weather(self, img):
        """分析天气条件（基于图像特征）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # 雾天特征：低对比度 + 较高亮度
        if std_brightness < 35 and mean_brightness > 100:
            weather = 'foggy'
        # 雨天特征（简化判断）
        elif std_brightness < 45 and mean_brightness < 120:
            weather = 'rainy'
        else:
            weather = 'clear'

        return weather

    def analyze_density(self, num_objects):
        """分析目标密度"""
        if num_objects <= 5:
            density = 'sparse'
        elif num_objects <= 10:
            density = 'medium'
        elif num_objects <= 20:
            density = 'dense'
        else:
            density = 'very_dense'
        return density

    def analyze_scale_distribution(self, annotations, img_h, img_w):
        """分析目标尺度分布"""
        small_count = 0
        medium_count = 0
        large_count = 0

        # 定义尺度阈值（相对于图像尺寸）
        img_area = img_h * img_w

        for ann in annotations:
            area_ratio = ann['size_pixels'] / img_area

            if area_ratio < 0.001:  # 小于0.1%图像面积
                small_count += 1
            elif area_ratio < 0.01:  # 小于1%图像面积
                medium_count += 1
            else:
                large_count += 1

        total = len(annotations)
        if total == 0:
            return 'mixed', {'small': 0, 'medium': 0, 'large': 0}

        small_ratio = small_count / total
        large_ratio = large_count / total

        if small_ratio > 0.6:
            scale = 'mostly_small'
        elif large_ratio > 0.4:
            scale = 'mostly_large'
        else:
            scale = 'mixed'

        return scale, {'small_ratio': small_ratio, 'medium_ratio': medium_count / total, 'large_ratio': large_ratio}

    def analyze_altitude(self, scale_info, density):
        """分析飞行高度"""
        if scale_info['large_ratio'] > 0.3 and density in ['sparse', 'medium']:
            altitude = 'low'
        elif scale_info['small_ratio'] > 0.5 and density in ['dense', 'very_dense']:
            altitude = 'high'
        else:
            altitude = 'medium'
        return altitude

    def analyze_occlusion(self, annotations):
        """分析遮挡程度"""
        if not annotations:
            return 'none'

        # 检查是否有遮挡信息
        occlusion_values = [ann.get('occlusion', 0) for ann in annotations]
        avg_occlusion = np.mean(occlusion_values) if occlusion_values else 0

        if avg_occlusion == 0:
            occlusion = 'none'
        elif avg_occlusion <= 1:
            occlusion = 'little'
        else:
            occlusion = 'heavy'

        return occlusion

    def analyze_image(self, img_path, annotations):
        """综合分析图像场景"""
        img = cv2.imread(img_path)
        if img is None:
            return {}

        h, w = img.shape[:2]

        # 光照分析
        illumination, illum_metrics = self.analyze_illumination(img)

        # 天气分析
        weather = self.analyze_weather(img)

        # 密度分析
        density = self.analyze_density(len(annotations))

        # 尺度分析
        scale, scale_metrics = self.analyze_scale_distribution(annotations, h, w)

        # 高度分析
        altitude = self.analyze_altitude(scale_metrics, density)

        # 遮挡分析
        occlusion = self.analyze_occlusion(annotations)

        # 组合场景标签
        scene_info = {
            'illumination': illumination,
            'weather': weather,
            'density': density,
            'scale': scale,
            'altitude': altitude,
            'occlusion': occlusion,
            'metrics': {
                **illum_metrics,
                **scale_metrics,
                'object_count': len(annotations)
            }
        }

        return scene_info


class SceneBucketManager:
    """场景分桶管理器：创建和管理场景分桶"""

    # [保持原有的SceneBucketManager类不变]
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.bucket_dir = os.path.join(output_dir, 'scene_buckets')
        os.makedirs(self.bucket_dir, exist_ok=True)

        self.buckets = defaultdict(list)
        self.combined_buckets = defaultdict(list)
        self.image_scene_map = {}

    def add_image(self, image_name, scene_info):
        """添加图像到场景分桶"""
        self.image_scene_map[image_name] = scene_info

        # 单维度分桶
        for dim, value in scene_info.items():
            if dim != 'metrics':  # 不把metrics作为分桶维度
                bucket_name = f"{dim}_{value}"
                self.buckets[bucket_name].append(image_name)

        # 组合场景分桶（高价值场景）
        metrics = scene_info.get('metrics', {})

        # 夜间密集场景
        if scene_info['illumination'] == 'night' and scene_info['density'] in ['dense', 'very_dense']:
            self.combined_buckets['night_dense'].append(image_name)

        # 雾天场景
        if scene_info['weather'] == 'foggy':
            self.combined_buckets['foggy'].append(image_name)

        # 小目标为主
        if scene_info['scale'] == 'mostly_small':
            self.combined_buckets['small_objects'].append(image_name)

        # 高空视角
        if scene_info['altitude'] == 'high':
            self.combined_buckets['high_altitude'].append(image_name)

        # 低空视角
        if scene_info['altitude'] == 'low':
            self.combined_buckets['low_altitude'].append(image_name)

        # 严重遮挡
        if scene_info['occlusion'] == 'heavy':
            self.combined_buckets['heavy_occlusion'].append(image_name)

    def save_buckets(self):
        """保存所有分桶文件"""
        # 保存单维度分桶
        for bucket_name, images in self.buckets.items():
            if len(images) >= 5:  # 只保存至少有5张图像的桶
                filepath = os.path.join(self.bucket_dir, f"{bucket_name}.txt")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(images))

        # 保存组合分桶
        for bucket_name, images in self.combined_buckets.items():
            if len(images) >= 5:
                filepath = os.path.join(self.bucket_dir, f"combined_{bucket_name}.txt")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(images))

    def generate_scene_report(self):
        """生成场景分桶报告"""
        report = {
            'total_images': len(self.image_scene_map),
            'bucket_summary': {},
            'combined_bucket_summary': {},
            'scene_distribution': defaultdict(lambda: defaultdict(int)),
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 统计各维度分布
        for image_name, scene_info in self.image_scene_map.items():
            for dim, value in scene_info.items():
                if dim != 'metrics':
                    report['scene_distribution'][dim][value] += 1

        # 转换defaultdict为普通dict
        report['scene_distribution'] = {
            dim: dict(counts) for dim, counts in report['scene_distribution'].items()
        }

        # 分桶大小统计
        for bucket_name, images in self.buckets.items():
            report['bucket_summary'][bucket_name] = len(images)

        for bucket_name, images in self.combined_buckets.items():
            report['combined_bucket_summary'][f"combined_{bucket_name}"] = len(images)

        # 保存报告
        report_path = os.path.join(self.bucket_dir, 'scene_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def visualize_distribution(self):
        """可视化场景分布"""
        if not self.image_scene_map:
            return

        # 统计各维度分布
        dims = ['illumination', 'weather', 'density', 'scale', 'altitude', 'occlusion']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, dim in enumerate(dims):
            counts = defaultdict(int)
            for info in self.image_scene_map.values():
                counts[info[dim]] += 1

            ax = axes[i]
            categories = list(counts.keys())
            values = list(counts.values())

            bars = ax.bar(categories, values, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
            ax.set_title(f'{dim.capitalize()} Distribution')
            ax.set_xlabel(dim)
            ax.set_ylabel('Count')

            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{value}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.bucket_dir, 'scene_distribution.png'), dpi=150)
        plt.close()


class AugmentationPipeline:
    """增强流水线，支持四类增强和开关配置"""

    def __init__(self, config_path='configs/augmentation_config.yaml'):
        """
    初始化增强流水线

    Args:
        config_path: 增强配置文件路径
    """
        # 如果配置文件不存在，使用默认配置
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        self.aug_config = self.config.get('augmentation', {})
        self.enabled = {}
        self.transforms = None

        # 构建增强流水线
        self._build_pipeline()

    def _get_default_config(self):
        """获取默认配置"""
        return {
            'global_probability': 0.5,
            'augmentation': {
                'photometric': {
                    'enabled': True,
                    'probability': 0.5,
                    'brightness_contrast': True,
                    'brightness_limit': 0.2,
                    'contrast_limit': 0.2,
                    'hue_saturation': True,
                    'hue_limit': 20,
                    'saturation_limit': 30,
                    'value_limit': 20,
                    'clahe': False,
                    'color_jitter': False,
                    'channel_shuffle': False,
                    'gamma': False
                },
                'geometric': {
                    'enabled': True,
                    'probability': 0.5,
                    'flip': True,
                    'vertical_flip': False,
                    'rotate': True,
                    'rotate_limit': 30,
                    'scale_translate': True,
                    'scale_range': [0.8, 1.2],
                    'translate_range': [-0.1, 0.1],
                    'shear': False,
                    'perspective': False,
                    'random_crop': False
                },
                'weather': {
                    'enabled': True,
                    'rain': True,
                    'rain_type': 'drizzle',
                    'rain_blur': 3,
                    'rain_brightness': 0.9,
                    'rain_prob': 0.3,
                    'fog': True,
                    'fog_lower': 0.3,
                    'fog_upper': 0.7,
                    'fog_alpha': 0.08,
                    'fog_prob': 0.3,
                    'snow': False,
                    'shadow': False,
                    'sun_flare': False
                },
                'motion': {
                    'enabled': True,
                    'probability': 0.3,
                    'motion_blur': True,
                    'blur_limit': 7,
                    'gaussian_blur': False,
                    'gaussian_noise': True,
                    'noise_var': [10.0, 50.0],
                    'iso_noise': False,
                    'defocus_blur': False,
                    'zoom_blur': False,
                    'pixel_dropout': False
                }
            }
        }

    def _build_pipeline(self):
        """根据配置构建增强流水线"""
        transform_list = []

        # 1. 光度增强 (Photometric)
        if self._is_enabled('photometric'):
            photo_config = self.aug_config.get('photometric', {})
            self.enabled['photometric'] = True
            transform_list.extend(self._build_photometric(photo_config))
        else:
            self.enabled['photometric'] = False

        # 2. 几何增强 (Geometric)
        if self._is_enabled('geometric'):
            geo_config = self.aug_config.get('geometric', {})
            self.enabled['geometric'] = True
            transform_list.extend(self._build_geometric(geo_config))
        else:
            self.enabled['geometric'] = False

        # 3. 天气增强 (Weather)
        if self._is_enabled('weather'):
            weather_config = self.aug_config.get('weather', {})
            self.enabled['weather'] = True
            transform_list.extend(self._build_weather(weather_config))
        else:
            self.enabled['weather'] = False

        # 4. 运动增强 (Motion)
        if self._is_enabled('motion'):
            motion_config = self.aug_config.get('motion', {})
            self.enabled['motion'] = True
            transform_list.extend(self._build_motion(motion_config))
        else:
            self.enabled['motion'] = False

        # 创建Compose对象
        if transform_list:
            self.transforms = A.Compose(
                transform_list,
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels'],
                    min_visibility=0.3,
                    min_area=16
                )
            )

    def _is_enabled(self, aug_type: str) -> bool:
        """检查增强类型是否开启"""
        return self.aug_config.get(aug_type, {}).get('enabled', False)

    def _build_photometric(self, config: dict) -> list:
        """构建光度增强"""
        transforms = []
        p = config.get('probability', 0.5)

        # 亮度对比度调整
        if config.get('brightness_contrast', True):
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=config.get('brightness_limit', 0.2),
                    contrast_limit=config.get('contrast_limit', 0.2),
                    p=p
                )
            )

        # 色调饱和度调整
        if config.get('hue_saturation', True):
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=config.get('hue_limit', 20),
                    sat_shift_limit=config.get('saturation_limit', 30),
                    val_shift_limit=config.get('value_limit', 20),
                    p=p
                )
            )

        # 直方图均衡化
        if config.get('clahe', False):
            transforms.append(
                A.CLAHE(
                    clip_limit=config.get('clahe_clip', 2.0),
                    tile_grid_size=config.get('clahe_grid', (8, 8)),
                    p=0.3
                )
            )

        # 色彩抖动
        if config.get('color_jitter', False):
            transforms.append(
                A.ColorJitter(
                    brightness=config.get('jitter_brightness', 0.2),
                    contrast=config.get('jitter_contrast', 0.2),
                    saturation=config.get('jitter_saturation', 0.2),
                    hue=config.get('jitter_hue', 0.1),
                    p=0.3
                )
            )

        # 通道随机打乱
        if config.get('channel_shuffle', False):
            transforms.append(A.ChannelShuffle(p=0.2))

        # 随机伽马校正
        if config.get('gamma', False):
            transforms.append(
                A.RandomGamma(
                    gamma_limit=config.get('gamma_limit', (80, 120)),
                    p=0.2
                )
            )

        return transforms

    def _build_geometric(self, config: dict) -> list:
        """构建几何增强"""
        transforms = []
        p = config.get('probability', 0.5)

        # 翻转
        if config.get('flip', True):
            transforms.append(A.HorizontalFlip(p=p))
            if config.get('vertical_flip', False):
                transforms.append(A.VerticalFlip(p=0.2))

        # 旋转
        if config.get('rotate', True):
            transforms.append(
                A.Rotate(
                    limit=config.get('rotate_limit', 30),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=p
                )
            )

        # 缩放和平移
        if config.get('scale_translate', True):
            transforms.append(
                A.Affine(
                    scale=config.get('scale_range', (0.8, 1.2)),
                    translate_percent=config.get('translate_range', (-0.1, 0.1)),
                    p=p
                )
            )

        # 错切
        if config.get('shear', False):
            transforms.append(
                A.Affine(
                    shear=config.get('shear_range', (-20, 20)),
                    p=0.3
                )
            )

        # 透视变换
        if config.get('perspective', False):
            transforms.append(
                A.Perspective(
                    scale=config.get('perspective_scale', 0.05),
                    p=0.2
                )
            )

        # 随机裁剪
        if config.get('random_crop', False):
            transforms.append(
                A.RandomResizedCrop(
                    height=config.get('crop_height', 640),
                    width=config.get('crop_width', 640),
                    scale=config.get('crop_scale', (0.7, 1.0)),
                    p=0.3
                )
            )

        return transforms

    def _build_weather(self, config: dict) -> list:
        """构建天气增强"""
        transforms = []

        # 雨
        if config.get('rain', False):
            transforms.append(
                A.RandomRain(
                    rain_type=config.get('rain_type', 'drizzle'),
                    blur_value=config.get('rain_blur', 3),
                    brightness_coefficient=config.get('rain_brightness', 0.9),
                    p=config.get('rain_prob', 0.3)
                )
            )

        # 雾
        if config.get('fog', False):
            transforms.append(
                A.RandomFog(
                    fog_coef_lower=config.get('fog_lower', 0.3),
                    fog_coef_upper=config.get('fog_upper', 0.7),
                    alpha_coef=config.get('fog_alpha', 0.08),
                    p=config.get('fog_prob', 0.3)
                )
            )

        # 雪
        if config.get('snow', False):
            transforms.append(
                A.RandomSnow(
                    snow_point_lower=config.get('snow_lower', 0.1),
                    snow_point_upper=config.get('snow_upper', 0.3),
                    brightness_coeff=config.get('snow_brightness', 2.5),
                    p=config.get('snow_prob', 0.2)
                )
            )

        # 阴影
        if config.get('shadow', False):
            transforms.append(
                A.RandomShadow(
                    shadow_roi=config.get('shadow_roi', (0, 0.5, 1, 1)),
                    num_shadows_lower=config.get('shadow_lower', 1),
                    num_shadows_upper=config.get('shadow_upper', 2),
                    shadow_dimension=config.get('shadow_dim', 5),
                    p=config.get('shadow_prob', 0.3)
                )
            )

        # 太阳耀斑
        if config.get('sun_flare', False):
            transforms.append(
                A.RandomSunFlare(
                    flare_roi=config.get('flare_roi', (0, 0, 1, 0.5)),
                    angle_lower=config.get('flare_angle_lower', 0),
                    angle_upper=config.get('flare_angle_upper', 1),
                    num_flare_circles_lower=config.get('flare_circles_lower', 6),
                    num_flare_circles_upper=config.get('flare_circles_upper', 10),
                    src_radius=config.get('flare_radius', 400),
                    src_color=config.get('flare_color', (255, 255, 255)),
                    p=config.get('flare_prob', 0.2)
                )
            )

        return transforms

    def _build_motion(self, config: dict) -> list:
        """构建运动增强"""
        transforms = []
        p = config.get('probability', 0.3)

        # 运动模糊
        if config.get('motion_blur', True):
            transforms.append(
                A.MotionBlur(
                    blur_limit=config.get('blur_limit', 7),
                    p=p
                )
            )

        # 高斯模糊
        if config.get('gaussian_blur', False):
            transforms.append(
                A.GaussianBlur(
                    blur_limit=config.get('gaussian_limit', (3, 7)),
                    p=0.3
                )
            )

        # 高斯噪声
        if config.get('gaussian_noise', True):
            transforms.append(
                A.GaussNoise(
                    var_limit=config.get('noise_var', (10.0, 50.0)),
                    p=p
                )
            )

        # ISO噪声
        if config.get('iso_noise', False):
            transforms.append(
                A.ISONoise(
                    color_shift=config.get('iso_color', (0.01, 0.05)),
                    intensity=config.get('iso_intensity', (0.1, 0.5)),
                    p=0.3
                )
            )

        # 散焦模糊
        if config.get('defocus_blur', False):
            transforms.append(
                A.Defocus(
                    radius=config.get('defocus_radius', (3, 5)),
                    alias_blur=config.get('defocus_alias', (0.3, 0.5)),
                    p=0.2
                )
            )

        # 缩放抖动
        if config.get('zoom_blur', False):
            transforms.append(
                A.ZoomBlur(
                    max_factor=config.get('zoom_factor', 1.2),
                    p=0.2
                )
            )

        # 像素级抖动
        if config.get('pixel_dropout', False):
            transforms.append(
                A.PixelDropout(
                    dropout_prob=config.get('dropout_prob', 0.01),
                    per_channel=config.get('dropout_per_channel', False),
                    p=0.2
                )
            )

        return transforms

    def __call__(self, image: np.ndarray, boxes: list, labels: list) -> tuple:
        """
    应用增强

    Args:
        image: 输入图像
        boxes: YOLO格式的边界框 [x_center, y_center, width, height]
        labels: 类别标签

    Returns:
        增强后的图像、边界框和标签
    """
        if self.transforms is None:
            return image, boxes, labels

        # 随机决定是否应用增强（整体开关）
        if random.random() < self.config.get('global_probability', 0.5):
            try:
                augmented = self.transforms(
                    image=image,
                    bboxes=boxes,
                    class_labels=labels
                )
                return augmented['image'], augmented['bboxes'], augmented['class_labels']
            except Exception as e:
                print(f"增强处理出错: {e}, 返回原图")
                return image, boxes, labels

        return image, boxes, labels

    def get_status(self) -> dict:
        """获取当前增强配置状态"""
        return {
            'enabled': self.enabled,
            'transform_count': len(self.transforms.transforms) if self.transforms else 0,
            'config': self.aug_config
        }

    def visualize_augmentations(self, image: np.ndarray, boxes: list, labels: list,
                                num_samples: int = 5, save_path: str = None):
        """可视化不同增强效果"""
        fig, axes = plt.subplots(1, num_samples + 1, figsize=(20, 4))

        # 显示原图
        orig_img = image.copy()
        # 绘制边界框
        h, w = orig_img.shape[:2]
        for box in boxes:
            x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')

        # 显示不同增强结果
        for i in range(num_samples):
            aug_img, aug_boxes, _ = self(image.copy(), boxes.copy(), labels.copy())

            # 绘制边界框
            h, w = aug_img.shape[:2]
            for box in aug_boxes:
                x_center, y_center, width, height = box
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                cv2.rectangle(aug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            axes[i + 1].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f'Aug {i + 1}')
            axes[i + 1].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存至: {save_path}")

        plt.close()


class AugmentationManager:
    """增强配置管理器，支持消融实验"""

    def __init__(self, config_dir: str = 'configs/ablation_configs'):
        self.config_dir = config_dir
        self.configs = {}
        self.pipelines = {}

        # 创建配置目录
        os.makedirs(config_dir, exist_ok=True)

        # 生成默认消融配置
        self._generate_ablation_configs()

        # 加载所有消融配置
        self._load_all_configs()

    def _generate_ablation_configs(self):
        """生成默认的消融实验配置"""
        # 基线配置（全部关闭）
        baseline = {
            'augmentation': {
                'photometric': {'enabled': False},
                'geometric': {'enabled': False},
                'weather': {'enabled': False},
                'motion': {'enabled': False}
            }
        }

        # 仅光度增强
        photometric_only = {
            'augmentation': {
                'photometric': {'enabled': True},
                'geometric': {'enabled': False},
                'weather': {'enabled': False},
                'motion': {'enabled': False}
            }
        }

        # 仅几何增强
        geometric_only = {
            'augmentation': {
                'photometric': {'enabled': False},
                'geometric': {'enabled': True},
                'weather': {'enabled': False},
                'motion': {'enabled': False}
            }
        }

        # 仅天气增强
        weather_only = {
            'augmentation': {
                'photometric': {'enabled': False},
                'geometric': {'enabled': False},
                'weather': {'enabled': True},
                'motion': {'enabled': False}
            }
        }

        # 仅运动增强
        motion_only = {
            'augmentation': {
                'photometric': {'enabled': False},
                'geometric': {'enabled': False},
                'weather': {'enabled': False},
                'motion': {'enabled': True}
            }
        }

        # 全部开启
        all_combined = {
            'augmentation': {
                'photometric': {'enabled': True},
                'geometric': {'enabled': True},
                'weather': {'enabled': True},
                'motion': {'enabled': True}
            }
        }

        # 保存配置
        configs = {
            'baseline': baseline,
            'photometric_only': photometric_only,
            'geometric_only': geometric_only,
            'weather_only': weather_only,
            'motion_only': motion_only,
            'all_combined': all_combined
        }

        for name, config in configs.items():
            config_path = os.path.join(self.config_dir, f'{name}.yaml')
            if not os.path.exists(config_path):
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True)

    def _load_all_configs(self):
        """加载所有消融配置"""
        if not os.path.exists(self.config_dir):
            return

        for config_file in os.listdir(self.config_dir):
            if config_file.endswith('.yaml'):
                config_name = os.path.splitext(config_file)[0]
                config_path = os.path.join(self.config_dir, config_file)

                with open(config_path, 'r', encoding='utf-8') as f:
                    self.configs[config_name] = yaml.safe_load(f)

    def get_pipeline(self, config_name: str) -> AugmentationPipeline:
        """获取指定配置的增强流水线"""
        if config_name not in self.pipelines:
            # 创建临时配置文件
            temp_config = self.configs[config_name].copy()

            # 保存临时配置
            temp_path = f'temp_{config_name}.yaml'
            with open(temp_path, 'w', encoding='utf-8') as f:
                yaml.dump(temp_config, f)

            # 创建流水线
            self.pipelines[config_name] = AugmentationPipeline(temp_path)

            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return self.pipelines[config_name]

    def get_config_summary(self) -> dict:
        """获取配置摘要"""
        summary = {}

        for config_name, config in self.configs.items():
            aug_config = config.get('augmentation', {})
            enabled = []

            for aug_type in ['photometric', 'geometric', 'weather', 'motion']:
                if aug_config.get(aug_type, {}).get('enabled', False):
                    enabled.append(aug_type)

            summary[config_name] = {
                'enabled': enabled,
                'total_enabled': len(enabled)
            }

        return summary


class UAVDatasetPreparer:
    def __init__(self, config_path='config.yaml'):
        """初始化数据集准备器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = self.config['raw_data_dir']
        self.output_dir = self.config['output_dir']
        self.class_mapping = self.config['class_mapping']

        # 初始化场景分析器和分桶管理器
        self.scene_analyzer = SceneAnalyzer()
        self.bucket_manager = SceneBucketManager(self.output_dir)

        # 初始化增强相关
        self.augmentation_config = self.config.get('augmentation', {})
        self.augmentation_enabled = self.augmentation_config.get('enabled', False)
        if self.augmentation_enabled:
            self.augmentation_pipeline = AugmentationPipeline('configs/augmentation_config.yaml')
        else:
            self.augmentation_pipeline = None

        # 创建输出目录
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'augmented_samples'), exist_ok=True)

        # 数据索引
        self.data_index = []
        self.image_scene_info = {}  # 存储每张图像的场景信息

    def apply_augmentation(self, image, boxes, labels, image_name):
        """应用数据增强并保存增强样本"""
        if not self.augmentation_enabled or self.augmentation_pipeline is None:
            return image, boxes, labels

        # 应用增强
        aug_image, aug_boxes, aug_labels = self.augmentation_pipeline(image, boxes, labels)

        # 随机保存一些增强样本用于可视化
        if random.random() < 0.01:  # 1%的概率保存
            sample_path = os.path.join(self.output_dir, 'augmented_samples', f'aug_{image_name}')
            # 绘制边界框
            h, w = aug_image.shape[:2]
            vis_img = aug_image.copy()
            for box in aug_boxes:
                x_center, y_center, width, height = box
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(sample_path, vis_img)

        return aug_image, aug_boxes, aug_labels

    def convert_visdrone_to_yolo(self):
        """
    转换VisDrone标注为YOLO格式，同时分析场景信息
    """
        print("正在转换VisDrone数据并分析场景...")

        # VisDrone类别映射
        visdrone_to_unified = {
            1: 'pedestrian', 2: 'person', 3: 'bicycle', 4: 'car',
            5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle',
            9: 'bus', 10: 'motor', 11: 'others'
        }

        # VisDrone数据路径
        visdrone_root = os.path.join(self.raw_dir, 'VisDrone2019-DET-train')
        img_dir = os.path.join(visdrone_root, 'images')
        anno_dir = os.path.join(visdrone_root, 'annotations')

        # 检查路径
        if not os.path.exists(img_dir):
            alt_img_dir = os.path.join(self.raw_dir, 'images')
            alt_anno_dir = os.path.join(self.raw_dir, 'annotations')
            if os.path.exists(alt_img_dir):
                img_dir = alt_img_dir
                anno_dir = alt_anno_dir
            else:
                print(f"错误: 找不到图像目录 {img_dir}")
                return

        img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
        img_files.extend(glob.glob(os.path.join(img_dir, '*.png')))

        print(f"找到 {len(img_files)} 张图像")

        converted_count = 0
        for img_path in tqdm(img_files, desc="处理VisDrone"):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            # 对应标注文件
            anno_path = os.path.join(anno_dir, f"{base_name}.txt")
            if not os.path.exists(anno_path):
                continue

            # 读取图像获取尺寸
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # 读取标注
            yolo_labels = []
            valid_boxes = []
            boxes_list = []  # 用于增强的边界框列表
            labels_list = []  # 用于增强的标签列表

            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue

                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])
                    score = float(parts[4])
                    category = int(parts[5])
                    occlusion = int(parts[7])  # VisDrone第8列是遮挡程度

                    if score == 0 or category not in visdrone_to_unified:
                        continue

                    unified_class = visdrone_to_unified[category]
                    if unified_class not in self.class_mapping:
                        continue
                    class_id = self.class_mapping[unified_class]

                    # 转换为YOLO格式
                    x_center = (bbox_left + bbox_width / 2) / w
                    y_center = (bbox_top + bbox_height / 2) / h
                    norm_width = bbox_width / w
                    norm_height = bbox_height / h

                    if (x_center <= 0 or x_center >= 1 or
                            y_center <= 0 or y_center >= 1 or
                            norm_width <= 0 or norm_width > 1 or
                            norm_height <= 0 or norm_height > 1):
                        continue

                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                    valid_boxes.append({
                        'class': unified_class,
                        'class_id': class_id,
                        'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                        'size_pixels': bbox_width * bbox_height,
                        'occlusion': occlusion
                    })
                    boxes_list.append([x_center, y_center, norm_width, norm_height])
                    labels_list.append(class_id)

            if len(yolo_labels) > 0:
                # 分析场景信息
                scene_info = self.scene_analyzer.analyze_image(img_path, valid_boxes)

                # 应用数据增强（如果开启）
                if self.augmentation_enabled and self.augmentation_pipeline:
                    aug_img, aug_boxes, aug_labels = self.apply_augmentation(
                        img, boxes_list, labels_list, img_name
                    )

                    # 如果有增强结果且框的数量变化不大，保存增强版本
                    if len(aug_boxes) > 0 and abs(len(aug_boxes) - len(boxes_list)) / len(boxes_list) < 0.3:
                        # 保存增强后的图像
                        aug_img_name = f"aug_{img_name}"
                        aug_img_path = os.path.join(self.output_dir, 'images', aug_img_name)
                        cv2.imwrite(aug_img_path, aug_img)

                        # 保存增强后的标注
                        aug_label_path = os.path.join(self.output_dir, 'labels', f"aug_{base_name}.txt")
                        with open(aug_label_path, 'w') as f:
                            for i, box in enumerate(aug_boxes):
                                f.write(f"{aug_labels[i]} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

                        # 添加到索引（可选，如果需要增强数据也加入训练集）
                        if self.augmentation_config.get('include_augmented', False):
                            for i, box in enumerate(aug_boxes):
                                self.data_index.append({
                                    'image_path': aug_img_path,
                                    'image_name': aug_img_name,
                                    'class': self._get_class_name(aug_labels[i]),
                                    'class_id': aug_labels[i],
                                    'bbox': str(box),
                                    'pixel_area': box[2] * box[3] * h * w,
                                    'source': 'VisDrone_aug',
                                    'scene_illumination': scene_info['illumination'],
                                    'scene_weather': scene_info['weather'],
                                    'scene_density': scene_info['density'],
                                    'scene_scale': scene_info['scale'],
                                    'scene_altitude': scene_info['altitude'],
                                    'scene_occlusion': scene_info['occlusion']
                                })

                # 复制原始图像
                dst_img_path = os.path.join(self.output_dir, 'images', img_name)
                shutil.copy(img_path, dst_img_path)

                # 保存原始标注
                label_path = os.path.join(self.output_dir, 'labels', f"{base_name}.txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))

                # 存储场景信息
                self.image_scene_info[img_name] = scene_info
                self.bucket_manager.add_image(img_name, scene_info)

                # 添加到索引
                for box in valid_boxes:
                    self.data_index.append({
                        'image_path': dst_img_path,
                        'image_name': img_name,
                        'class': box['class'],
                        'class_id': box['class_id'],
                        'bbox': str(box['bbox']),
                        'pixel_area': box['size_pixels'],
                        'occlusion': box['occlusion'],
                        'source': 'VisDrone',
                        'scene_illumination': scene_info['illumination'],
                        'scene_weather': scene_info['weather'],
                        'scene_density': scene_info['density'],
                        'scene_scale': scene_info['scale'],
                        'scene_altitude': scene_info['altitude'],
                        'scene_occlusion': scene_info['occlusion']
                    })

                converted_count += 1

        print(f"VisDrone转换完成: 成功转换 {converted_count} 张图像")

    def _get_class_name(self, class_id):
        """根据class_id获取类别名称"""
        for name, cid in self.class_mapping.items():
            if cid == class_id:
                return name
        return 'unknown'

    def convert_uavdt_to_yolo(self):
        """
    转换UAVDT标注为YOLO格式
    """
        uavdt_path = os.path.join(self.raw_dir, 'UAVDT')
        if not os.path.exists(uavdt_path):
            return

        print("正在转换UAVDT数据...")

        uavdt_to_unified = {
            1: 'car', 2: 'truck', 3: 'bus'
        }

        video_dir = os.path.join(uavdt_path, 'data')
        anno_dir = os.path.join(uavdt_path, 'annotations')

        if not os.path.exists(video_dir):
            return

        videos = os.listdir(video_dir)
        converted_count = 0

        for video in tqdm(videos, desc="转换UAVDT"):
            video_path = os.path.join(video_dir, video)
            anno_path = os.path.join(anno_dir, f"{video}.txt")

            if not os.path.isdir(video_path) or not os.path.exists(anno_path):
                continue

            # 读取标注
            frame_annotations = defaultdict(list)
            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue

                    frame_idx = int(parts[0])
                    bbox_left = float(parts[2])
                    bbox_top = float(parts[3])
                    bbox_width = float(parts[4])
                    bbox_height = float(parts[5])
                    category = int(parts[8])

                    frame_annotations[frame_idx].append({
                        'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                        'category': category
                    })

            # 处理每一帧
            for frame_idx, annotations in frame_annotations.items():
                img_name = f"{frame_idx:06d}.jpg"
                img_path = os.path.join(video_path, img_name)

                if not os.path.exists(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                yolo_labels = []
                valid_boxes = []
                boxes_list = []
                labels_list = []

                for ann in annotations:
                    category = ann['category']
                    if category not in uavdt_to_unified:
                        continue

                    unified_class = uavdt_to_unified[category]
                    if unified_class not in self.class_mapping:
                        continue
                    class_id = self.class_mapping[unified_class]

                    bbox_left, bbox_top, bbox_width, bbox_height = ann['bbox']

                    x_center = (bbox_left + bbox_width / 2) / w
                    y_center = (bbox_top + bbox_height / 2) / h
                    norm_width = bbox_width / w
                    norm_height = bbox_height / h

                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                    valid_boxes.append({
                        'class': unified_class,
                        'class_id': class_id,
                        'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                        'size_pixels': bbox_width * bbox_height
                    })
                    boxes_list.append([x_center, y_center, norm_width, norm_height])
                    labels_list.append(class_id)

                if len(yolo_labels) > 0:
                    # 分析场景
                    scene_info = self.scene_analyzer.analyze_image(img_path, valid_boxes)

                    # 保存图像
                    unique_name = f"uavdt_{video}_{img_name}"
                    dst_img_path = os.path.join(self.output_dir, 'images', unique_name)
                    shutil.copy(img_path, dst_img_path)

                    # 保存标注
                    label_path = os.path.join(self.output_dir, 'labels',
                                              f"{os.path.splitext(unique_name)[0]}.txt")
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_labels))

                    # 存储场景信息
                    self.image_scene_info[unique_name] = scene_info
                    self.bucket_manager.add_image(unique_name, scene_info)

                    # 添加到索引
                    for box in valid_boxes:
                        self.data_index.append({
                            'image_path': dst_img_path,
                            'image_name': unique_name,
                            'class': box['class'],
                            'class_id': box['class_id'],
                            'bbox': str(box['bbox']),
                            'pixel_area': box['size_pixels'],
                            'source': 'UAVDT',
                            'video': video,
                            'frame': frame_idx,
                            'scene_illumination': scene_info['illumination'],
                            'scene_weather': scene_info['weather'],
                            'scene_density': scene_info['density'],
                            'scene_scale': scene_info['scale'],
                            'scene_altitude': scene_info['altitude'],
                            'scene_occlusion': scene_info['occlusion']
                        })

                    converted_count += 1

        print(f"UAVDT转换完成: 成功转换 {converted_count} 张图像")

    def analyze_data(self):
        """分析数据统计信息"""
        if not self.data_index:
            return None

        df = pd.DataFrame(self.data_index)

        stats = {
            'total_annotations': len(df),
            'total_images': df['image_path'].nunique(),
            'class_distribution': df['class'].value_counts().to_dict(),
            'source_distribution': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'avg_annotations_per_image': len(df) / df['image_path'].nunique() if df['image_path'].nunique() > 0 else 0
        }

        # 目标尺寸统计
        df['is_small'] = df['pixel_area'] < 1024
        df['is_medium'] = (df['pixel_area'] >= 1024) & (df['pixel_area'] < 96 * 96)
        df['is_large'] = df['pixel_area'] >= 96 * 96

        stats['small_target_ratio'] = float(df['is_small'].mean())
        stats['medium_target_ratio'] = float(df['is_medium'].mean())
        stats['large_target_ratio'] = float(df['is_large'].mean())

        # 场景统计
        scene_columns = ['scene_illumination', 'scene_weather', 'scene_density',
                         'scene_scale', 'scene_altitude', 'scene_occlusion']

        stats['scene_statistics'] = {}
        for col in scene_columns:
            if col in df.columns:
                stats['scene_statistics'][col] = df[col].value_counts().to_dict()

        # 增强统计
        if 'source' in df.columns:
            aug_count = len(df[df['source'].str.contains('aug', na=False)])
            stats['augmented_samples'] = int(aug_count)

        # 保存统计结果
        with open(os.path.join(self.output_dir, 'data_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats

    def generate_data_index(self):
        """生成数据索引文件"""
        if not self.data_index:
            print("警告: 没有数据生成索引")
            return

        df = pd.DataFrame(self.data_index)

        # 保存详细索引
        csv_path = os.path.join(self.output_dir, 'data_index.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"数据索引已保存至: {csv_path}")
        print(f"总记录数: {len(df)}")

        # 生成图像列表
        image_paths = df['image_path'].unique()
        with open(os.path.join(self.output_dir, 'image_list.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(image_paths))

        return df

    def generate_augmentation_report(self):
        """生成增强配置报告"""
        if not self.augmentation_enabled or not self.augmentation_pipeline:
            return

        status = self.augmentation_pipeline.get_status()

        report = {
            'augmentation_enabled': self.augmentation_enabled,
            'enabled_types': [k for k, v in status['enabled'].items() if v],
            'transform_count': status['transform_count'],
            'config': status['config'],
            'include_augmented_in_dataset': self.augmentation_config.get('include_augmented', False)
        }

        # 保存报告
        report_path = os.path.join(self.output_dir, 'augmentation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"增强报告已保存至: {report_path}")

    def run(self):
        """执行完整的数据准备流程"""
        print("=" * 60)
        print("无人机航拍数据集准备（含场景分桶和数据增强）")
        print("=" * 60)
        print(f"原始数据目录: {self.raw_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"数据增强: {'开启' if self.augmentation_enabled else '关闭'}")
        print("=" * 60)

        # 转换数据
        self.convert_visdrone_to_yolo()
        self.convert_uavdt_to_yolo()

        if self.data_index:
            # 生成数据索引
            df = self.generate_data_index()

            # 保存场景分桶
            print("\n生成场景分桶...")
            self.bucket_manager.save_buckets()

            # 生成场景报告
            scene_report = self.bucket_manager.generate_scene_report()
            self.bucket_manager.visualize_distribution()

            # 生成增强报告
            self.generate_augmentation_report()

            # 分析数据
            stats = self.analyze_data()

            if stats:
                print("\n" + "=" * 60)
                print("数据统计:")
                print(f"总标注数: {stats['total_annotations']}")
                print(f"总图像数: {stats['total_images']}")
                print(f"平均每图标注数: {stats['avg_annotations_per_image']:.1f}")
                print(f"小目标比例: {stats['small_target_ratio']:.2%}")
                print(f"中目标比例: {stats['medium_target_ratio']:.2%}")
                print(f"大目标比例: {stats['large_target_ratio']:.2%}")

                if 'augmented_samples' in stats:
                    print(f"增强样本数: {stats['augmented_samples']}")

                print("\n场景分布摘要:")
                print(f"  光照: {dict(scene_report['scene_distribution'].get('illumination', {}))}")
                print(f"  天气: {dict(scene_report['scene_distribution'].get('weather', {}))}")
                print(f"  密度: {dict(scene_report['scene_distribution'].get('density', {}))}")

                print("\n分桶统计:")
                important_buckets = ['illumination_night', 'weather_foggy', 'density_dense',
                                     'scale_mostly_small', 'altitude_high', 'occlusion_heavy']
                for bucket in important_buckets:
                    if bucket in scene_report['bucket_summary']:
                        print(f"  {bucket}: {scene_report['bucket_summary'][bucket]} 张")

                print("\n组合场景分桶:")
                for bucket, count in scene_report['combined_bucket_summary'].items():
                    if count > 0:
                        print(f"  {bucket}: {count} 张")
        else:
            print("\n警告: 没有成功转换任何数据！")

        print("\n" + "=" * 60)
        print("数据集准备完成！")
        print(f"输出目录结构:")
        print(f"  {self.output_dir}/")
        print(f"    ├── images/          # 所有图像")
        print(f"    ├── labels/          # YOLO格式标注")
        print(f"    ├── augmented_samples/ # 增强样本可视化")
        print(f"    ├── scene_buckets/   # 场景分桶文件")
        print(f"    │   ├── illumination_day.txt")
        print(f"    │   ├── illumination_night.txt")
        print(f"    │   ├── weather_foggy.txt")
        print(f"    │   ├── density_dense.txt")
        print(f"    │   ├── combined_small_objects.txt")
        print(f"    │   ├── combined_night_dense.txt")
        print(f"    │   ├── scene_report.json")
        print(f"    │   └── scene_distribution.png")
        print(f"    ├── data_index.csv   # 详细索引")
        print(f"    ├── image_list.txt   # 图像路径列表")
        print(f"    ├── data_stats.json  # 数据统计")
        print(f"    └── augmentation_report.json  # 增强配置报告")
        print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='无人机航拍数据集准备（场景分桶+数据增强）')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--raw_dir', type=str, help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='./yolo_output', help='输出目录')
    parser.add_argument('--aug_enabled', action='store_true', help='开启数据增强')
    parser.add_argument('--aug_config', type=str, default='configs/augmentation_config.yaml',
                        help='增强配置文件路径')

    args = parser.parse_args()

    # 您的数据路径
    YOUR_DATA_PATH = r'E:/servicecomp/数据0/data'
    YOUR_OUTPUT_PATH = r'E:/servicecomp/数据0/yolo_output'

    # 创建配置文件
    def load_config(config_path='config.yaml'):
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config


    # 加载配置
    config = load_config('config.yaml')

    config_path = args.config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    print(f"配置文件已创建: {config_path}")
    print(f"原始数据目录: {config['raw_data_dir']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"数据增强: {'开启' if config['augmentation']['enabled'] else '关闭'}")

    # 验证路径
    visdrone_path = os.path.join(config['raw_data_dir'], 'VisDrone2019-DET-train')
    img_path = os.path.join(visdrone_path, 'images')

    print(f"\n检查路径:")
    print(f"VisDrone根目录: {visdrone_path}")
    print(f"是否存在: {os.path.exists(visdrone_path)}")
    print(f"Images目录: {img_path}")
    print(f"是否存在: {os.path.exists(img_path)}")

    if os.path.exists(img_path):
        files = glob.glob(os.path.join(img_path, '*.jpg'))[:5]
        if files:
            print(f"找到图像文件示例: {[os.path.basename(f) for f in files]}")
    else:
        print("错误: 找不到图像目录，请检查路径！")
        exit(1)

    # 运行数据转换
    preparer = UAVDatasetPreparer(config_path)
    preparer.run()