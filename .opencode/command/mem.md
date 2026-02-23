---
description: Show memory statistics
---

Run the following Python code to get memory stats:

```python
import sys
sys.path.insert(0, '/home/raman/repos/Trading-Algo')
from utils.memory.memory_commands import handle_mem_stats
result = handle_mem_stats()
print(result)
```
