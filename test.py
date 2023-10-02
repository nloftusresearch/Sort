import numpy as np

object_type_distributions = []
for obj_class in range(5):
    object_type_distributions.append(np.empty((1, 5)))

for o in object_type_distributions:
    print(o)
    print()
