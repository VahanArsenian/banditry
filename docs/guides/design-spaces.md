# Design spaces

A `DesignSpace` describes the parameters an agent searches over. It is built
from a list of spec dicts, one per parameter:

```python
from banditry import DesignSpace

space = DesignSpace.parse([
    {"name": "learning_rate", "type": "num",  "lb": 1e-4, "ub": 1e-1},
    {"name": "num_layers",    "type": "int",  "lb": 1,    "ub": 8},
    {"name": "use_dropout",   "type": "bool"},
    {"name": "optimiser",     "type": "cat",  "categories": ["adam", "sgd", "rmsprop"]},
])
```

## Parameter types

| `type` | Spec keys | Domain |
|---|---|---|
| `"num"` | `lb`, `ub` | continuous float in `[lb, ub]` |
| `"int"` | `lb`, `ub` | integer in `[lb, ub]` (inclusive) |
| `"bool"` | — | `True` / `False` |
| `"cat"` | `categories` | one of the listed values |

Suggestions come back as a pandas DataFrame with one column per parameter in
the **original (untransformed) domain** — `optimiser` is `"adam"`, not an
index; `use_dropout` is a bool.

!!! note "Column ordering"
    `space.para_names` lists numeric-ish parameters first (`num`, `int`,
    `bool`), then categorical ones. Suggestion DataFrames follow this order.

## Transforms

Internally, models see a transformed representation: numeric-ish parameters
as a float tensor, categorical parameters as integer category indices (fed to
embeddings or one-hot encodings by the surrogates).

```python
x_num, x_cat = space.transform(df)      # DataFrame -> (FloatTensor, LongTensor)
df_back = space.inverse_transform(x_num, x_cat)
```

## Custom parameter types

Register a subclass of `Parameter` under a new `type` name to extend the
spec format:

```python
from banditry import DesignSpace
from banditry.variable_domains import NumericParameter

class LogUniformParameter(NumericParameter):
    ...  # override transform / inverse_transform / sample

DesignSpace.register_parameter_type("lognum", LogUniformParameter)
space = DesignSpace.parse([{"name": "lr", "type": "lognum", "lb": 1e-5, "ub": 1e-1}])
```

Full API: [Design spaces reference](../api/variable-domains.md).
