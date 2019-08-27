# Locked HashMap for Rust

This is HashMap implementation that can be shared by multi thread programing easy,
because of internal mutability.

## Example

```rust
use locked_hash::LockedHashMap;

// no `mut` keyword
let db = LockedHashMap::new();

crossbeam::scope(|scope| {
    for _ in 0..threads {
        // spawn some thread with sharing mutable HashMap
        scope.spawn(|_| {
            worker(&db);
        });
    }
})
```
