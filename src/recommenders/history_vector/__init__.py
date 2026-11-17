"""History-vector recommenders — importing this package registers MLP, MLPDeep, NCF and NCFGMF."""

from __future__ import annotations

from src.recommenders.history_vector import mlp as _mlp  # noqa: F401
from src.recommenders.history_vector import mlp_deep as _mlp_deep  # noqa: F401
from src.recommenders.history_vector import ncf as _ncf  # noqa: F401
from src.recommenders.history_vector import ncf_gmf as _ncf_gmf  # noqa: F401
