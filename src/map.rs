use std::borrow::Borrow;
use std::collections::{hash_map::RandomState, HashMap};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::RwLock;

use num_cpus;

/// Locked Hash Map
///
///
pub struct LockedHashMap<K, V, S1 = RandomState, S2 = RandomState> {
    key_hash_builder: S1,
    data: Vec<RwLock<HashMap<K, V, S2>>>,
}

const DEFAULT_LOCK_DIVIDED: usize = 256;

impl<K, V> LockedHashMap<K, V, RandomState, RandomState>
where
    K: Hash + Eq,
    V: Clone,
{
    /// Create an almost empty `LockedHashMap`.
    ///
    /// The default lock-divided is `256`, it means `LockedHashMap` has
    /// `cpus * 256` lock in data.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    /// let map: LockedHashMap<i64, String> = LockedHashMap::new();
    /// ```
    #[inline]
    pub fn new() -> LockedHashMap<K, V, RandomState, RandomState> {
        Self::allocate(
            num_cpus::get() * DEFAULT_LOCK_DIVIDED,
            0,
            RandomState::new(),
            RandomState::new(),
        )
    }

    /// Create an empty `LockedHashMap` with specified lock-divide and capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    /// let map: LockedHashMap<i64, String> = LockedHashMap::with_div_and_capacity(16, 100);
    /// ```
    #[inline]
    pub fn with_div_and_capacity(
        div: usize,
        capacity: usize,
    ) -> LockedHashMap<K, V, RandomState, RandomState> {
        Self::allocate(div, capacity, RandomState::new(), RandomState::new())
    }
}

impl<K, V> Default for LockedHashMap<K, V, RandomState, RandomState>
where
    K: Hash + Eq,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S1, S2> LockedHashMap<K, V, S1, S2> {
    /// Return the approximate total capacity of `LockedHashMap`.
    ///
    /// This tales time `O(div)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    /// let map: LockedHashMap<i64, String> = LockedHashMap::with_div_and_capacity(16, 100);
    /// assert!(map.capacity() >=16 * 100);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.iter().map(|h| h.read().unwrap().capacity()).sum()
    }

    /// Return the approximate total number of elements `LockedHashMap` has.
    ///
    /// This tales `O(div)` time.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    ///
    /// let map = LockedHashMap::with_div_and_capacity(16, 100);
    /// assert_eq!(map.len(), 0);
    ///
    /// map.insert(1, 10);
    /// assert_eq!(map.len(), 1)
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.data.iter().map(|h| h.read().unwrap().len()).sum()
    }

    /// Return `true` if `LockedHashMap` is empty.
    ///
    /// This tales `O(div)` time.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    ///
    /// let map = LockedHashMap::new();
    /// assert!(map.is_empty());
    ///
    /// map.insert(1, 10);
    /// assert!(!map.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V, S1, S2> LockedHashMap<K, V, S1, S2>
where
    K: Eq + Hash,
    V: Clone,
    S1: BuildHasher,
    S2: BuildHasher + Clone,
{
    /// Allocate heap memory.
    #[inline]
    fn allocate(
        div: usize,
        capacity: usize,
        key_hash_builder: S1,
        map_hash_builder: S2,
    ) -> LockedHashMap<K, V, S1, S2> {
        let mut data = Vec::with_capacity(div);
        for _ in 0..div {
            data.push(RwLock::new(HashMap::with_capacity_and_hasher(
                capacity,
                map_hash_builder.clone(),
            )));
        }
        LockedHashMap {
            key_hash_builder,
            data,
        }
    }

    /// Calculate a position of lock corresponding to the key.
    #[inline]
    fn key_pos<Q>(&self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut hasher = self.key_hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.data.len()
    }

    /// Create an empty `LockedHashMap` which will use the given hash builder to hash keys.
    ///
    /// `key_hash_builder` is used to calculate lock position.
    /// `map_hash_builder` is used to `HashMap`.
    #[inline]
    pub fn with_hasher(key_hash_builder: S1, map_hash_builder: S2) -> LockedHashMap<K, V, S1, S2> {
        Self::allocate(
            num_cpus::get() * DEFAULT_LOCK_DIVIDED,
            0,
            key_hash_builder,
            map_hash_builder,
        )
    }

    /// Create an empty `LockedHashMap` with specified lock-divide, capacity and hasher.
    ///
    /// View also [`with_hasher`].
    ///
    /// [`with_hasher`]: #method.with_hasher
    #[inline]
    pub fn with_div_and_capacity_and_hasher(
        div: usize,
        capacity: usize,
        key_hash_builder: S1,
        map_hash_builder: S2,
    ) -> LockedHashMap<K, V, S1, S2> {
        Self::allocate(div, capacity, key_hash_builder, map_hash_builder)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// This method does *not* take `&mut self`, but `&self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    ///
    /// // No `mut` keyword
    /// let map = LockedHashMap::new();
    /// assert_eq!(map.insert(10, "a"), None);
    /// assert_eq!(map.len(), 1);
    ///
    /// map.insert(42, "b");
    /// assert_eq!(map.insert(42, "c"), Some("b"));
    /// assert_eq!(map.len(), 2);
    /// ```
    #[inline]
    pub fn insert(&self, k: K, v: V) -> Option<V> {
        let mut map = self.data[self.key_pos(&k)].write().unwrap();
        map.insert(k, v)
    }

    /// Return a cloned value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    ///
    /// let map = LockedHashMap::new();
    /// map.insert(10, 1);
    /// // Return value is not reference
    /// assert_eq!(map.get(&10), Some(1));
    /// assert_eq!(map.get(&20), None)
    /// ```
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let map = self.data[self.key_pos(k)].read().unwrap();
        map.get(k).and_then(|v| Some(v.clone()))
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    ///
    /// let map = LockedHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let map = self.data[self.key_pos(k)].read().unwrap();
        map.contains_key(k)
    }
}

pub struct IntoIter<K, V, S> {
    data: Vec<RwLock<HashMap<K, V, S>>>,
    iter: std::vec::IntoIter<(K, V)>,
}

impl<K, V, S> Iterator for IntoIter<K, V, S> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.iter.next() {
            Some(v)
        } else {
            while let Some(lock) = self.data.pop() {
                self.iter = lock
                    .into_inner()
                    .unwrap()
                    .drain()
                    .collect::<Vec<_>>()
                    .into_iter();
                if let Some(v) = self.iter.next() {
                    return Some(v);
                }
            }
            None
        }
    }
}

impl<K, V, S1, S2> IntoIterator for LockedHashMap<K, V, S1, S2> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, S2>;

    /// Return an iterator that moves out of a map.
    /// That Iterator visiting all key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use locked_hash::LockedHashMap;
    ///
    /// let map = LockedHashMap::new();
    /// map.insert(1, "a");
    /// let mut iter = map.into_iter();
    /// assert_eq!(iter.next(), Some((1, "a")));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    fn into_iter(self) -> IntoIter<K, V, S2> {
        IntoIter {
            // TODO optimize
            data: self.data,
            iter: Vec::new().into_iter(),
        }
    }
}
