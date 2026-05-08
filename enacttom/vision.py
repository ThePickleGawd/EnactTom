from __future__ import annotations

import re
import json
import base64
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - optional Habitat/vision dependency
    torch = None


def pil_image_to_data_url(image: Image.Image, fmt: str = "png") -> str:
    buffered = io.BytesIO()
    save_fmt = "JPEG" if fmt.lower() in ("jpg", "jpeg") else fmt.upper()
    image.save(buffered, format=save_fmt)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


@dataclass
class VisualFrameHandle:
    frame_id: str
    agent_id: str
    turn: int
    frame_index: int
    skill_step: int
    sim_step: int
    kind: str
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_image_array(image: Any) -> Optional[np.ndarray]:
    """Convert a tensor/array-like RGB observation into a uint8 HWC array."""
    if image is None:
        return None

    if torch is not None and torch.is_tensor(image):
        image_arr = image.detach().cpu().numpy()
    else:
        image_arr = np.asarray(image)

    if image_arr.ndim == 4:
        image_arr = image_arr[0]
    if image_arr.ndim == 3 and image_arr.shape[0] in (1, 3, 4):
        image_arr = np.transpose(image_arr, (1, 2, 0))
    if image_arr.ndim != 3:
        return None

    if image_arr.shape[-1] == 1:
        image_arr = np.repeat(image_arr, 3, axis=-1)
    elif image_arr.shape[-1] > 3:
        image_arr = image_arr[..., :3]

    if image_arr.dtype != np.uint8:
        max_val = float(np.max(image_arr)) if image_arr.size else 0.0
        if max_val <= 1.0:
            image_arr = np.clip(image_arr * 255.0, 0, 255).astype(np.uint8)
        else:
            image_arr = np.clip(image_arr, 0, 255).astype(np.uint8)

    return image_arr


def extract_agent_rgb_frame(
    observations: Dict[str, Any],
    agent_id: str,
) -> Optional[np.ndarray]:
    """Extract the best-effort first-person RGB frame for one agent."""
    if not isinstance(observations, dict):
        return None

    rgb_items = [(key, value) for key, value in observations.items() if "rgb" in key.lower()]
    if not rgb_items:
        return None

    preferred_prefix = f"{agent_id}_"
    preferred = [item for item in rgb_items if item[0].startswith(preferred_prefix)]
    if preferred:
        preferred.sort(key=lambda item: item[0])
        return normalize_image_array(preferred[0][1])

    # Single-agent configs often expose only one RGB key with no agent prefix.
    rgb_items.sort(key=lambda item: item[0])
    return normalize_image_array(rgb_items[0][1])


class VisualObservationStore:
    """Persist per-turn RGB observations and return lightweight frame handles."""

    def __init__(self, root_dir: str, image_format: str = "jpeg"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.image_format = image_format.lower().lstrip(".") or "png"
        self._frame_counts: Dict[tuple[int, str], int] = {}
        self._handles_by_turn: Dict[int, Dict[str, List[VisualFrameHandle]]] = {}

    def capture(
        self,
        observations: Dict[str, Any],
        agent_ids: Iterable[str],
        turn: int,
        skill_step: int,
        sim_step: int,
        kind: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        captured: Dict[str, List[Dict[str, Any]]] = {}

        for agent_id in agent_ids:
            frame = extract_agent_rgb_frame(observations, agent_id)
            if frame is None:
                continue

            key = (turn, agent_id)
            frame_index = self._frame_counts.get(key, 0)
            self._frame_counts[key] = frame_index + 1

            agent_dir = self.root_dir / agent_id / f"turn_{turn:04d}"
            agent_dir.mkdir(parents=True, exist_ok=True)

            frame_id = f"{agent_id}_t{turn:04d}_f{frame_index:04d}"
            filename = (
                f"{frame_id}_s{max(0, int(skill_step)):04d}_sim{max(0, int(sim_step)):06d}"
                f"_{kind}.{self.image_format}"
            )
            frame_path = agent_dir / filename
            Image.fromarray(frame).save(frame_path)

            handle = VisualFrameHandle(
                frame_id=frame_id,
                agent_id=agent_id,
                turn=int(turn),
                frame_index=int(frame_index),
                skill_step=int(skill_step),
                sim_step=int(sim_step),
                kind=str(kind),
                path=str(frame_path),
            )
            self._handles_by_turn.setdefault(int(turn), {}).setdefault(agent_id, []).append(handle)
            captured.setdefault(agent_id, []).append(handle.to_dict())

        return captured

    def get_turn_handles(self, turn: int, agent_id: str) -> List[Dict[str, Any]]:
        handles = self._handles_by_turn.get(int(turn), {}).get(agent_id, [])
        return [handle.to_dict() for handle in handles]

    def export_index(self, path: Optional[str] = None) -> str:
        index_path = Path(path) if path else self.root_dir / "index.json"
        payload: Dict[str, Any] = {"turns": {}}

        for turn, per_agent in sorted(self._handles_by_turn.items()):
            payload["turns"][str(turn)] = {
                agent_id: [handle.to_dict() for handle in handles]
                for agent_id, handles in sorted(per_agent.items())
            }

        index_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return str(index_path)


def build_candidate_frame_set(
    handles: Sequence[Dict[str, Any]],
    max_candidates: int,
) -> List[Dict[str, Any]]:
    """Downsample a large frame set while preserving chronology and the final frame."""
    if max_candidates <= 0:
        return []

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for handle in handles:
        frame_id = handle.get("frame_id")
        if not frame_id or frame_id in seen:
            continue
        seen.add(frame_id)
        deduped.append(handle)

    if len(deduped) <= max_candidates:
        return deduped

    indices = np.linspace(0, len(deduped) - 1, num=max_candidates, dtype=int)
    selected: List[Dict[str, Any]] = []
    used = set()
    for idx in indices.tolist():
        if idx in used:
            continue
        used.add(idx)
        selected.append(deduped[idx])

    if deduped[-1]["frame_id"] not in {item["frame_id"] for item in selected}:
        selected[-1] = deduped[-1]

    return selected


def parse_selector_response(
    response_text: str,
    candidate_handles: Sequence[Dict[str, Any]],
    min_select: int,
    max_select: int,
) -> List[Dict[str, Any]]:
    """Resolve selector text into a validated ordered subset of candidate handles."""
    if not candidate_handles:
        return []

    response_text = response_text or ""
    positions = []
    for handle in candidate_handles:
        frame_id = handle.get("frame_id", "")
        if not frame_id:
            continue
        match = re.search(rf"(?<![\w-]){re.escape(frame_id)}(?![\w-])", response_text)
        if match:
            positions.append((match.start(), frame_id))

    selected_ids = [frame_id for _, frame_id in sorted(positions)]

    if not selected_ids:
        selected_ids = [candidate_handles[-1].get("frame_id", "")]

    ordered: List[Dict[str, Any]] = []
    seen = set()
    for frame_id in selected_ids:
        if not frame_id or frame_id in seen:
            continue
        seen.add(frame_id)
        for handle in candidate_handles:
            if handle.get("frame_id") == frame_id:
                ordered.append(handle)
                break
        if len(ordered) >= max_select:
            break

    if len(ordered) < min_select:
        for handle in reversed(candidate_handles):
            frame_id = handle.get("frame_id")
            if not frame_id or frame_id in seen:
                continue
            ordered.insert(0, handle)
            seen.add(frame_id)
            if len(ordered) >= min_select:
                break

    return ordered[:max_select]


def load_frame_as_data_url(handle: Dict[str, Any]) -> str:
    """Load a saved frame from disk and return a data URL for multimodal LLM input."""
    path = handle["path"]
    image = Image.open(path).convert("RGB")
    ext = Path(path).suffix.lower().lstrip(".")
    fmt = "jpeg" if ext in ("jpg", "jpeg") else ext or "png"
    return pil_image_to_data_url(image, fmt=fmt)
