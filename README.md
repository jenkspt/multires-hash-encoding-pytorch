# Neural Hash Encoding

This is an unofficial pytorch implementation of the key datastructures from [Instant Neural Graphics Primitives](https://github.com/NVlabs/instant-ngp)

```
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```

For an example of how to create a drop in replacement for standard NeRF models, take a look at:
- [https://github.com/jenkspt/NeuS/tree/hash](https://github.com/jenkspt/NeuS/tree/hash)
- [https://github.com/jenkspt/NeuS/blob/hash/models/hash_fields.py](https://github.com/jenkspt/NeuS/blob/hash/models/hash_fields.py)