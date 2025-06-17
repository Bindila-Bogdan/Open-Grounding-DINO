import argparse
import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from torchmetrics.detection import MeanAveragePrecision


def get_box_coordinates(H, W, boxes, device):
    updated_boxes = []

    for box in boxes:
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        updated_boxes.append(box)

    return updated_boxes


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def get_grounding_output(
    model,
    image,
    caption,
    box_threshold,
    text_threshold=None,
    with_logits=True,
    cpu_only=False,
    token_spans=None,
):
    assert (
        text_threshold is not None or token_spans is not None
    ), "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt), token_span=token_spans
        ).to(
            image.device
        )  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for token_span, logit_phr in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = " ".join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend(
                    [phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num]
                )
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


def load_annotations(annotations_path):
    painting_annotations = []

    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                painting_annotations.append(json.loads(stripped_line))

    return painting_annotations


def get_labels_to_ids(painting_annotations):
    unique_annotations = set()

    for painting_annotation in painting_annotations:
        unique_annotations.update(painting_annotation["grounding"]["caption"][:-2].split(" . "))

    labels_to_ids = dict(zip(unique_annotations, list(range(len(unique_annotations)))))

    return labels_to_ids


def get_bounding_boxes(painting_annotation, pred, size, boxes_filt, labels_to_ids, device):
    predicted_bboxes = {
        "boxes": torch.stack(get_box_coordinates(size[1], size[0], boxes_filt, device)),
        "scores": torch.tensor(
            [float(label.split("(")[1][:-1]) for label in pred["labels"]], device=device
        ),
        "labels": torch.tensor(
            [
                (
                    labels_to_ids[label]
                    if label in labels_to_ids.keys()
                    else max(labels_to_ids.values()) + 1
                )
                for label in [label.split("(")[0] for label in pred["labels"]]
            ],
            device=device,
        ),
    }

    if len(painting_annotation["grounding"]["regions"]) != 0:
        target_bboxes = {
            "boxes": torch.tensor(
                [annotation["bbox"] for annotation in painting_annotation["grounding"]["regions"]],
                device=device,
            ),
            "labels": torch.tensor(
                [
                    (
                        labels_to_ids[label]
                        if label in labels_to_ids.keys()
                        else max(labels_to_ids.values()) + 1
                    )
                    for label in [
                        annotation["phrase"]
                        for annotation in painting_annotation["grounding"]["regions"]
                    ]
                ],
                device=device,
            ),
        }
    else:
        # treat the case when for an image there's no ground truth
        target_bboxes = {
            "boxes": torch.empty((0, 4)).to(device),
            "labels": torch.empty((0,), dtype=torch.int64).to(device),
        }

    return predicted_bboxes, target_bboxes


def compute_mean_average_precision(predictions, targets, device, show_map_per_class=False):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True).to(device)

    metric.update(predictions, targets)
    metrics = metric.compute()

    map_50 = float(metrics["map_50"])
    map_50_95 = float(metrics["map"])

    print(f"mAP@50: {map_50}")
    print(f"mAP@50-95: {map_50_95}")

    # if performance per class is available, show it
    if show_map_per_class:
        try:
            print(f"mAP per class: {metrics['map_per_class']}")
            print(f"classes: {metrics['classes']}")
        except:
            pass

    return map_50, map_50_95


def evaluate_model(
    config_file,
    checkpoint_path,
    annotations_file,
    images_dir,
    logging_path,
    test,
    store_annotated_images=False,
    box_threshold=0.34,
    text_threshold=0.32,
    cpu_only=False,
):
    if not cpu_only:
        device = "cuda"
    else:
        device = "cpu"

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)

    all_predicted_bboxes = []
    all_ground_truth_bboxes = []

    # load annotations
    painting_annotations = load_annotations(annotations_file)
    labels_to_ids = get_labels_to_ids(painting_annotations)

    for painting_annotation in painting_annotations:
        image_name = painting_annotation["filename"]
        input_labels = painting_annotation["grounding"]["caption"][:-2].split(" . ")

        # load image and get individual labels
        image_pil, image = load_image(images_dir + image_name)

        all_boxes_filt = []
        pred_phrases = []

        for input_label in input_labels:
            # run model
            current_boxes_filt, current_pred_phrases = get_grounding_output(
                model,
                image,
                input_label + " .",
                box_threshold,
                text_threshold,
                cpu_only=cpu_only,
                token_spans=None,
            )
            all_boxes_filt.append(current_boxes_filt)
            pred_phrases.extend(current_pred_phrases)

        boxes_filt = torch.cat(all_boxes_filt, dim=0)

        # post-process bounding boxes
        pred = {
            "boxes": boxes_filt,
            "size": [image_pil.size[1], image_pil.size[0]],
            "labels": pred_phrases,
        }
        pred_bboxes, target_bboxes = get_bounding_boxes(
            painting_annotation, pred, image_pil.size, boxes_filt, labels_to_ids, device
        )

        all_predicted_bboxes.append(pred_bboxes)
        all_ground_truth_bboxes.append(target_bboxes)

        # save annotated image
        if store_annotated_images:
            plot_boxes_to_image(image_pil, pred)[0].save(f"./pred_{image_name}")

    map_50, map_50_95 = compute_mean_average_precision(
        all_predicted_bboxes, all_ground_truth_bboxes, device
    )
    map_values = {"map_50": map_50, "map_50_95": map_50_95}

    log_stats = {**{f"test_{k}": v for k, v in map_values.items()}}

    if test:
        file_name = "evaluation"
        log_stats["weights"] = checkpoint_path
    else:
        file_name = "intermediate_evaluation"

    with open(f"{logging_path}/{file_name}.json", "w") as f:
        json.dump(log_stats, f, indent=4)

    return map_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO evaluation", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--annotations_file",
        "-a",
        type=str,
        required=True,
        help="path to the file with annotations",
    )
    parser.add_argument(
        "--images_dir", "-d", type=str, required=True, help="directory where images are located"
    )
    parser.add_argument(
        "--store_annotated_images",
        "-s",
        action="store_true",
        help="store annotated images, default=False",
    )
    parser.add_argument(
        "--logging_path", "-l", type=str, required=True, help="path where the results to be stored"
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="True if the results are obtained after training"
    )
    parser.add_argument("--box_threshold", type=float, default=0.34, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.32, help="text threshold")
    parser.add_argument(
        "--cpu_only", action="store_true", help="running on cpu only!, default=False"
    )
    args = parser.parse_args()

    evaluate_model(
        args.config_file,
        args.checkpoint_path,
        args.annotations_file,
        args.images_dir,
        args.logging_path,
        args.test,
        args.store_annotated_images,
        args.box_threshold,
        args.text_threshold,
        args.cpu_only,
    )
