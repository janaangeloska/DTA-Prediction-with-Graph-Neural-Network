import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import torch

# Residues of the structure the GNN branch is built from, so both arms see the same protein.
DEFAULT_SEQUENCE_SOURCE = "pdb"

# Keeps a dimension that is constant across proteins at zero instead of dividing by zero.
STANDARDIZE_MIN_STD = 1e-6

# ProtTrans vocabularies have no tokens for these; the Rostlab model cards map them to X.
PROTTRANS_RARE_RESIDUES = re.compile(r"[UZOB]")


def window_starts(length: int, window: int) -> list[int]:
    # Half-window stride, plus a final window flush with the end so no residue is left out.
    starts = list(range(0, length - window + 1, window // 2))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


class EmbeddingAdapter(ABC):
    def __init__(self, embedding_dim: int, max_residues: int | None) -> None:
        self.embedding_dim = embedding_dim
        self.max_residues = max_residues
        self.device: torch.device | None = None

    @abstractmethod
    def load(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def _per_residue(self, sequence: str) -> torch.Tensor:
        pass

    def needs_windows(self, sequence: str) -> bool:
        return self.max_residues is not None and len(sequence) > self.max_residues

    def embed(self, sequence: str) -> torch.Tensor:
        if self.device is None:
            raise RuntimeError("Call load() before embed()")
        if not self.needs_windows(sequence):
            return self._embed_window(sequence)
        total = torch.zeros(len(sequence), self.embedding_dim)
        counts = torch.zeros(len(sequence), 1)
        for start in window_starts(len(sequence), self.max_residues):
            end = start + self.max_residues
            total[start:end] += self._embed_window(sequence[start:end])
            counts[start:end] += 1
        return total / counts

    def _embed_window(self, sequence: str) -> torch.Tensor:
        with torch.no_grad():
            embedding = self._per_residue(sequence).float().cpu()
        expected = (len(sequence), self.embedding_dim)
        if tuple(embedding.shape) != expected:
            raise ValueError(
                f"Expected embedding of shape {expected}, got {tuple(embedding.shape)}"
            )
        return embedding


class TransformersAdapter(EmbeddingAdapter):
    def __init__(
        self,
        checkpoint: str,
        embedding_dim: int,
        max_residues: int | None,
        model_class: str,
        tokenizer_class: str,
        leading_special_tokens: int,
        prottrans_input: bool,
    ) -> None:
        super().__init__(embedding_dim, max_residues)
        self.checkpoint = checkpoint
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.leading_special_tokens = leading_special_tokens
        self.prottrans_input = prottrans_input

    def load(self, device: torch.device) -> None:
        # Imported here so baseline training does not depend on transformers.
        import transformers

        tokenizer_kwargs = {"do_lower_case": False} if self.prottrans_input else {}
        self.tokenizer = getattr(transformers, self.tokenizer_class).from_pretrained(
            self.checkpoint, **tokenizer_kwargs
        )
        model = getattr(transformers, self.model_class).from_pretrained(self.checkpoint)
        self.model = model.to(device).eval()
        self.device = device

    def _per_residue(self, sequence: str) -> torch.Tensor:
        text = sequence
        if self.prottrans_input:
            text = " ".join(PROTTRANS_RARE_RESIDUES.sub("X", sequence))
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        hidden = self.model(**inputs).last_hidden_state[0]
        start = self.leading_special_tokens
        return hidden[start : start + len(sequence)]


class ESMCAdapter(EmbeddingAdapter):
    # 2048-token context minus the BOS and EOS tokens.
    MAX_RESIDUES = 2046

    def __init__(self, model_name: str, embedding_dim: int) -> None:
        super().__init__(embedding_dim, self.MAX_RESIDUES)
        self.model_name = model_name

    def load(self, device: torch.device) -> None:
        # Imported here so baseline training does not depend on the esm package.
        from esm.models.esmc import ESMC

        self.model = ESMC.from_pretrained(self.model_name).to(device).eval()
        self.device = device

    def _per_residue(self, sequence: str) -> torch.Tensor:
        from esm.sdk.api import ESMProtein, LogitsConfig

        protein = self.model.encode(ESMProtein(sequence=sequence))
        output = self.model.logits(
            protein, LogitsConfig(sequence=True, return_embeddings=True)
        )
        return output.embeddings[0, 1:-1]


ADAPTERS: dict[str, Callable[[], EmbeddingAdapter]] = {
    "esmc_300m": lambda: ESMCAdapter("esmc_300m", embedding_dim=960),
    "esmc_600m": lambda: ESMCAdapter("esmc_600m", embedding_dim=1152),
    "prot_t5_xl_uniref50": lambda: TransformersAdapter(
        "Rostlab/prot_t5_xl_uniref50",
        embedding_dim=1024,
        max_residues=None,
        model_class="T5EncoderModel",
        tokenizer_class="T5Tokenizer",
        leading_special_tokens=0,
        prottrans_input=True,
    ),
    "esm2_650m": lambda: TransformersAdapter(
        "facebook/esm2_t33_650M_UR50D",
        embedding_dim=1280,
        max_residues=1022,
        model_class="EsmModel",
        tokenizer_class="AutoTokenizer",
        leading_special_tokens=1,
        prottrans_input=False,
    ),
    "protbert": lambda: TransformersAdapter(
        "Rostlab/prot_bert",
        embedding_dim=1024,
        max_residues=39998,
        model_class="BertModel",
        tokenizer_class="BertTokenizer",
        leading_special_tokens=1,
        prottrans_input=True,
    ),
}

MODEL_NAMES = tuple(sorted(ADAPTERS))


def build_adapter(model_name: str) -> EmbeddingAdapter:
    if model_name not in ADAPTERS:
        raise ValueError(
            f"Unknown model {model_name!r}, valid names: {', '.join(MODEL_NAMES)}"
        )
    return ADAPTERS[model_name]()


def mean_pool(per_residue: torch.Tensor) -> torch.Tensor:
    return per_residue.mean(dim=0)


def cache_path(cache_dir: str, model_name: str, sequence: str, kind: str) -> str:
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    return os.path.join(cache_dir, model_name, f"{digest}.{kind}.pt")


def pooled_path(cache_dir: str, model_name: str, sequence: str) -> str:
    return cache_path(cache_dir, model_name, sequence, "pooled")


def residues_path(cache_dir: str, model_name: str, sequence: str) -> str:
    return cache_path(cache_dir, model_name, sequence, "residues")


def is_cached(
    cache_dir: str, model_name: str, sequence: str, keep_residues: bool
) -> bool:
    if not os.path.exists(pooled_path(cache_dir, model_name, sequence)):
        return False
    return not keep_residues or os.path.exists(
        residues_path(cache_dir, model_name, sequence)
    )


def save_once(tensor: torch.Tensor, path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, path)


def cache_embedding(
    adapter: EmbeddingAdapter,
    cache_dir: str,
    model_name: str,
    sequence: str,
    keep_residues: bool,
) -> None:
    if is_cached(cache_dir, model_name, sequence, keep_residues):
        return
    per_residue_file = residues_path(cache_dir, model_name, sequence)
    if os.path.exists(per_residue_file):
        per_residue = torch.load(
            per_residue_file, map_location="cpu", weights_only=True
        )
    else:
        per_residue = adapter.embed(sequence)
    if keep_residues:
        save_once(per_residue, per_residue_file)
    save_once(mean_pool(per_residue), pooled_path(cache_dir, model_name, sequence))


def windowed_path(cache_dir: str, model_name: str, sequence_source: str) -> str:
    return os.path.join(cache_dir, model_name, f"windowed_{sequence_source}.json")


def write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def sequences_path(cache_dir: str, sequence_source: str) -> str:
    return os.path.join(cache_dir, f"sequences_{sequence_source}.json")


def pdb_sizes(
    pdb_file: Callable[[str], str], names: Iterable[str]
) -> dict[str, int]:
    # Re-parsing every PDB at load time takes about a minute; file sizes catch a replaced file.
    return {name: os.path.getsize(pdb_file(name)) for name in names}


def write_sequences(
    cache_dir: str,
    sequence_source: str,
    sequences: dict[str, str],
    source_files: dict[str, int],
) -> None:
    write_json(
        {"sequences": sequences, "source_files": source_files},
        sequences_path(cache_dir, sequence_source),
    )


def read_sequences(
    cache_dir: str, sequence_source: str, pdb_file: Callable[[str], str]
) -> dict[str, str]:
    path = sequences_path(cache_dir, sequence_source)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No sequence list at {path}. Run compute_protein_embeddings.py first."
        )
    with open(path) as file:
        record = json.load(file)
    if sequence_source == "pdb":
        current = pdb_sizes(pdb_file, record["sequences"])
        for name, size in record["source_files"].items():
            if current[name] != size:
                raise ValueError(
                    f"{pdb_file(name)} differs from the file {path} was built "
                    f"from ({current[name]} vs {size} bytes). Rerun "
                    "compute_protein_embeddings.py with the same --pdb-set so both arms "
                    "use the same structures."
                )
    return record["sequences"]


def load_pooled_embeddings(
    cache_dir: str,
    model_name: str,
    pdb_file: Callable[[str], str],
    sequence_source: str = DEFAULT_SEQUENCE_SOURCE,
    standardize: bool = False,
) -> dict[str, torch.Tensor]:
    embedding_dim = build_adapter(model_name).embedding_dim
    sequences = read_sequences(cache_dir, sequence_source, pdb_file)
    missing = [
        name
        for name, sequence in sequences.items()
        if not os.path.exists(pooled_path(cache_dir, model_name, sequence))
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(sequences)} proteins have no {model_name} "
            f"embedding in {cache_dir}, for example {missing[0]}. Run "
            f"compute_protein_embeddings.py --model {model_name} "
            f"--sequence-source {sequence_source}."
        )
    # Davis variants share their wild-type structure, so many names map to one sequence.
    pooled_by_sequence: dict[str, torch.Tensor] = {}
    for sequence in sorted(set(sequences.values())):
        path = pooled_path(cache_dir, model_name, sequence)
        pooled = torch.load(path, map_location="cpu", weights_only=True)
        assert tuple(pooled.shape) == (embedding_dim,), (
            f"{path} holds shape {tuple(pooled.shape)}, expected ({embedding_dim},)"
        )
        pooled_by_sequence[sequence] = pooled
    if standardize:
        pooled_by_sequence = standardize_vectors(pooled_by_sequence)
    return {name: pooled_by_sequence[seq] for name, seq in sequences.items()}


def standardize_vectors(vectors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Uses embeddings only, never labels.
    stacked = torch.stack(list(vectors.values()))
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp_min(STANDARDIZE_MIN_STD)
    return {key: (vector - mean) / std for key, vector in vectors.items()}
