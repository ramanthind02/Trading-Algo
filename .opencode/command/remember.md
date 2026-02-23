---
description: Store text to persistent memory
---

Run the following Python code to store text to memory:

```python
import sys
sys.path.insert(0, '/home/raman/repos/Trading-Algo')
from utils.memory.memory_commands import handle_remember
result = handle_remember("""$ARGUMENTS""")
print(result)
```
