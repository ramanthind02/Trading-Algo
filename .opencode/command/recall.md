---
description: Retrieve relevant memory
---

Run the following Python code to recall from memory:

```python
import sys
sys.path.insert(0, '/home/raman/repos/Trading-Algo')
from utils.memory.memory_commands import handle_recall
result = handle_recall("""$ARGUMENTS""")
print(result)
```
