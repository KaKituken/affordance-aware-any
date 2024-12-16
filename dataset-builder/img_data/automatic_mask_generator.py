# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from segment_anything.modeling import Sam
from segment_anything.predictor import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crops_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crops_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        for i in range(len(mask_data)):
            if self.min_mask_region_area > 0:
                mask_data[i] = self.postprocess_small_regions(
                    mask_data[i],
                    self.min_mask_region_area,
                    max(self.box_nms_thresh, self.crop_nms_thresh),
                )

        # Encode masks
            if self.output_mode == "coco_rle":
                mask_data[i]["segmentations"] = [coco_encode_rle(rle) for rle in mask_data[i]["rles"]]
            elif self.output_mode == "binary_mask":
                mask_data[i]["segmentations"] = [rle_to_mask(rle) for rle in mask_data[i]["rles"]]
            else:
                mask_data[i]["segmentations"] = mask_data[i]["rles"]

        # Write mask records
        curr_anns_list = []
        for i in range(3):
            curr_anns_list.append([])
            for idx in range(len(mask_data[i]["segmentations"])):
                ann = {
                    "segmentation": mask_data[i]["segmentations"][idx],
                    "area": area_from_rle(mask_data[i]["rles"][idx]),
                    "bbox": box_xyxy_to_xywh(mask_data[i]["boxes"][idx]).tolist(),
                    "predicted_iou": mask_data[i]["iou_preds"][idx].item(),
                    "point_coords": [mask_data[i]["points"][idx].tolist()],
                    "stability_score": mask_data[i]["stability_score"][idx].item(),
                    "crop_box": box_xyxy_to_xywh(mask_data[i]["crop_boxes"][idx]).tolist(),
                }
                curr_anns_list[i].append(ann)

        return curr_anns_list

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data_list = []
        for i in range(3):
            data_list.append(MaskData())
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            for i in range(len(crop_data)):
                data_list[i].cat(crop_data[i])

        # Remove duplicate masks between crops
        for i in range(len(data_list)):
            if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
                scores = 1 / box_area(data_list[i]["crop_boxes"])
                scores = scores.to(data_list[i]["boxes"].device)
                keep_by_nms = batched_nms(
                    data_list[i]["boxes"].float(),
                    scores,
                    torch.zeros(len(data_list[i]["boxes"])),  # categories
                    iou_threshold=self.crop_nms_thresh,
                )
                data_list[i].filter(keep_by_nms)

            data_list[i].to_numpy()
        return data_list

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data_list = []
        for i in range(3):
            data_list.append(MaskData())
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            for i in range(len(batch_data)):
                data_list[i].cat(batch_data[i])
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        for i in range(len(data_list)):
            keep_by_nms = batched_nms(
            data_list[i]["boxes"].float(),
            data_list[i]["iou_preds"],
            torch.zeros(len(data_list[i]["boxes"])),  # categories
            iou_threshold=self.box_nms_thresh,
            )
            data_list[i].filter(keep_by_nms)

        # Return to the original image frame
            data_list[i]["boxes"] = uncrop_boxes_xyxy(data_list[i]["boxes"], crop_box)
            data_list[i]["points"] = uncrop_points(data_list[i]["points"], crop_box)
            data_list[i]["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data_list[i]["rles"]))])

        return data_list

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data_list = []
        for i in range(masks.shape[1]):
            data_list.append(MaskData(
                masks = masks[:,i,:,:],
                iou_preds = iou_preds[:,i],
                points=torch.as_tensor(points)
            ))
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            for i in range(len(data_list)):
                if i == 0:
                    iou_thresh = self.pred_iou_thresh
                else:
                    iou_thresh = self.pred_iou_thresh
                keep_mask = data_list[i]['iou_preds'] > iou_thresh
                data_list[i].filter(keep_mask)

        # Calculate stability score
        for i in range(len(data_list)):
            data_list[i]['stability_score'] = calculate_stability_score(
                data_list[i]['masks'],self.predictor.model.mask_threshold,self.stability_score_offset
            )
        if self.stability_score_thresh > 0.0:
            for i in range(len(data_list)):
                if i == 0:
                    stability_score = self.stability_score_thresh
                else:
                    stability_score = self.stability_score_thresh
                keep_mask = data_list[i]['stability_score'] >= stability_score
                data_list[i].filter(keep_mask)

        # Threshold masks and calculate boxes
        for i in range(len(data_list)):
            data_list[i]["masks"] = data_list[i]["masks"] > self.predictor.model.mask_threshold
            data_list[i]["boxes"] = batched_mask_to_box(data_list[i]["masks"])

        # Filter boxes that touch crop boundaries
            keep_mask = ~is_box_near_crop_edge(data_list[i]["boxes"], crop_box, [0, 0, orig_w, orig_h])
            if not torch.all(keep_mask):
                data_list[i].filter(keep_mask)

        # Compress to RLE
            data_list[i]["masks"] = uncrop_masks(data_list[i]["masks"], crop_box, orig_h, orig_w)
            data_list[i]["rles"] = mask_to_rle_pytorch(data_list[i]["masks"])
            del data_list[i]["masks"]

        return data_list

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros(len(boxes)),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
