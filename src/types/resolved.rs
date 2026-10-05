use core::fmt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use miniscript::iter::{Tree, TreeLike};

use super::{
    EnumInfo, StructuralType, TypeConstructible, TypeDeconstructible, TypeInner, UIntType,
};
use crate::num::NonZeroPow2Usize;

/// SimplicityHL type without type aliases.
///
/// Types built from aliases share their parts, so walking them as trees can take time
/// exponential in the number of aliases. Walks over types visit each shared part once.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct ResolvedType {
    inner: TypeInner<Arc<Self>>,
    /// Whether the type mentions `!` outside enum payloads.
    has_never: bool,
    /// Whether the type mentions an enum.
    has_enum: bool,
}

impl ResolvedType {
    /// Create a type from its outermost constructor,
    /// recording what its parts mention so that checking it takes constant time.
    fn new(inner: TypeInner<Arc<Self>>) -> Self {
        let (has_never, has_enum) = match &inner {
            TypeInner::Never => (true, false),
            TypeInner::Enum(_) => (false, true),
            TypeInner::Boolean | TypeInner::UInt(_) => (false, false),
            TypeInner::Option(part) | TypeInner::Array(part, _) | TypeInner::List(part, _) => {
                Self::mentions([part])
            }
            TypeInner::Either(left, right) => Self::mentions([left, right]),
            TypeInner::Tuple(elements) => Self::mentions(elements.iter()),
        };

        Self {
            inner,
            has_never,
            has_enum,
        }
    }

    /// Whether any part mentions `!` outside enum payloads, and whether any mentions an enum.
    fn mentions<'a>(parts: impl IntoIterator<Item = &'a Arc<Self>>) -> (bool, bool) {
        parts
            .into_iter()
            .fold((false, false), |(never, enumeration), part| {
                (never || part.has_never, enumeration || part.has_enum)
            })
    }

    /// Access the inner type primitive.
    pub fn as_inner(&self) -> &TypeInner<Arc<Self>> {
        &self.inner
    }

    /// Call `visit` on each part of the type one level down, including the payloads of an enum.
    fn for_each_part<'a>(&'a self, mut visit: impl FnMut(&'a Self)) {
        match self.as_inner() {
            TypeInner::Either(left, right) => {
                visit(left);
                visit(right);
            }
            TypeInner::Option(part) | TypeInner::Array(part, _) | TypeInner::List(part, _) => {
                visit(part);
            }
            TypeInner::Tuple(elements) => {
                for element in elements.iter() {
                    visit(element);
                }
            }
            TypeInner::Enum(info) => {
                for variant in info.variants() {
                    visit(variant.payload_type());
                }
            }
            TypeInner::Boolean | TypeInner::UInt(_) | TypeInner::Never => {}
        }
    }

    /// The number of parts of the type one level down, as written in its name.
    fn n_parts(&self) -> usize {
        match self.as_inner() {
            TypeInner::Either(..) => 2,
            TypeInner::Option(_) | TypeInner::Array(..) | TypeInner::List(..) => 1,
            TypeInner::Tuple(elements) => elements.len(),
            TypeInner::Boolean | TypeInner::UInt(_) | TypeInner::Enum(_) | TypeInner::Never => 0,
        }
    }
}

/// Nominal enum types.
///
/// These methods are inherent rather than part of [`TypeConstructible`] and [`TypeDeconstructible`].
/// Those traits model the structural type algebra that every type universe (aliased, resolved, structural)
/// shares, while a nominal enum exists only at the resolved level.
///
/// At the structural level its identity is erased into a balanced sum, and at the source level enums
/// enter types by name only.
/// Keeping the constructor off the shared traits also means that only [`crate::ast`]'s scope
/// (which owns the uniqueness of declaration ids) can mint enum types.
impl ResolvedType {
    /// Create a nominal enum type from the given definition.
    pub const fn enumeration(info: EnumInfo) -> Self {
        Self {
            inner: TypeInner::Enum(info),
            has_never: false,
            has_enum: true,
        }
    }

    /// Access the enum definition if this is an enum type.
    pub const fn as_enum(&self) -> Option<&EnumInfo> {
        match &self.inner {
            TypeInner::Enum(info) => Some(info),
            _ => None,
        }
    }

    /// Check whether the type mentions an enum, at any nesting depth.
    pub fn contains_enum(&self) -> bool {
        self.has_enum
    }
}

/// The uninhabited type.
impl ResolvedType {
    /// Create the uninhabited type.
    pub const fn never() -> Self {
        Self {
            inner: TypeInner::Never,
            has_never: true,
            has_enum: false,
        }
    }

    /// Check whether this is the uninhabited type.
    pub const fn is_never(&self) -> bool {
        matches!(self.inner, TypeInner::Never)
    }

    /// Check whether the type mentions the uninhabited type, at any nesting depth
    /// except inside enum payloads, because messages show an enum by its name.
    pub(crate) fn contains_never(&self) -> bool {
        self.has_never
    }

    /// Check whether the type can be lowered to a structural type.
    ///
    /// Unlike [`Self::contains_never`], this looks inside enum payloads.
    pub(crate) fn has_structural_type(&self) -> bool {
        // Only enums can hide `!`, so only parts with enums are searched, each once.
        let mut visited = HashSet::new();
        let mut stack = vec![self];

        while let Some(ty) = stack.pop() {
            if ty.has_never {
                return false;
            }
            if ty.has_enum && visited.insert(std::ptr::from_ref(ty)) {
                ty.for_each_part(|part| stack.push(part));
            }
        }

        true
    }

    /// Check whether the types are equal, where `!` is equal to every type.
    ///
    /// During analysis, `!` stands for a broken type whose error was already reported,
    /// so use this instead of `==` wherever a mismatch would be reported as an error.
    /// Enums are compared by identity, so `!` inside an enum payload is not looked at.
    pub(crate) fn compatible(&self, other: &Self) -> bool {
        // Each pair of parts is compared once, without recursion, so that shared and deep
        // types stay cheap. A pair that is met again is skipped: it is still waiting to be
        // compared, or it was compatible, because the first mismatch ends the search.
        let mut seen = HashSet::new();
        let mut stack = vec![(self, other)];

        while let Some((a, b)) = stack.pop() {
            let pair = (std::ptr::from_ref(a), std::ptr::from_ref(b));
            if std::ptr::eq(a, b) || !seen.insert(pair) {
                continue;
            }

            match (a.as_inner(), b.as_inner()) {
                (TypeInner::Never, _) | (_, TypeInner::Never) => {}
                (TypeInner::Either(a1, a2), TypeInner::Either(b1, b2)) => {
                    stack.push((a1.as_ref(), b1.as_ref()));
                    stack.push((a2.as_ref(), b2.as_ref()));
                }
                (TypeInner::Option(a1), TypeInner::Option(b1)) => {
                    stack.push((a1.as_ref(), b1.as_ref()));
                }
                // Copies of the same alias share their elements.
                (TypeInner::Tuple(a1), TypeInner::Tuple(b1)) if Arc::ptr_eq(a1, b1) => {}
                (TypeInner::Tuple(a1), TypeInner::Tuple(b1)) if a1.len() == b1.len() => {
                    for (x, y) in a1.iter().zip(b1.iter()) {
                        stack.push((x.as_ref(), y.as_ref()));
                    }
                }
                (TypeInner::Array(a1, m), TypeInner::Array(b1, n)) if m == n => {
                    stack.push((a1.as_ref(), b1.as_ref()));
                }
                (TypeInner::List(a1, m), TypeInner::List(b1, n)) if m == n => {
                    stack.push((a1.as_ref(), b1.as_ref()));
                }
                (TypeInner::Boolean, TypeInner::Boolean) => {}
                (TypeInner::UInt(m), TypeInner::UInt(n)) if m == n => {}
                (TypeInner::Enum(m), TypeInner::Enum(n)) if m == n => {}
                _ => return false,
            }
        }

        true
    }
}

/// The number of parts after which a type in a message is cut off.
///
/// Types that share their parts, like `type A1 = (A0, A0); type A2 = (A1, A1);`,
/// can be exponentially longer written out than declared.
const MESSAGE_PARTS: usize = 1000;

/// Types in messages.
impl ResolvedType {
    /// Display the type for a message, cut off after [`MESSAGE_PARTS`] parts.
    ///
    /// `!` is shown as `_`, because users cannot write it:
    /// it stands for a type whose error was reported already.
    pub(crate) fn in_message(&self) -> impl fmt::Display + '_ {
        InMessage(self)
    }
}

struct InMessage<'a>(&'a ResolvedType);

impl fmt::Display for InMessage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut n_written = 0;
        // Types whose closing text is not written yet.
        let mut open: Vec<&ResolvedType> = Vec::new();

        for data in self.0.verbose_pre_order_iter() {
            let ty = data.node;
            if data.n_children_yielded == 0 {
                n_written += 1;
                if MESSAGE_PARTS < n_written {
                    f.write_str("...")?;
                    // Close what is open, so that the message still reads like a type.
                    for open_ty in open.iter().rev() {
                        open_ty.inner.display(f, open_ty.n_parts())?;
                    }
                    return Ok(());
                }
                if ty.is_never() {
                    f.write_str("_")?;
                    continue;
                }
            }

            ty.inner.display(f, data.n_children_yielded)?;
            let n_parts = ty.n_parts();
            if 0 < n_parts && data.n_children_yielded == 0 {
                open.push(ty);
            } else if 0 < n_parts && data.n_children_yielded == n_parts {
                open.pop();
            }
        }

        Ok(())
    }
}

impl TypeConstructible for ResolvedType {
    fn either(left: Self, right: Self) -> Self {
        Self::new(TypeInner::Either(Arc::new(left), Arc::new(right)))
    }

    fn option(inner: Self) -> Self {
        Self::new(TypeInner::Option(Arc::new(inner)))
    }

    fn boolean() -> Self {
        Self::new(TypeInner::Boolean)
    }

    fn tuple<I: IntoIterator<Item = Self>>(elements: I) -> Self {
        Self::new(TypeInner::Tuple(
            elements.into_iter().map(Arc::new).collect(),
        ))
    }

    fn array(element: Self, size: usize) -> Self {
        Self::new(TypeInner::Array(Arc::new(element), size))
    }

    fn list(element: Self, bound: NonZeroPow2Usize) -> Self {
        Self::new(TypeInner::List(Arc::new(element), bound))
    }
}

impl TypeDeconstructible for ResolvedType {
    fn as_either(&self) -> Option<(&Self, &Self)> {
        match self.as_inner() {
            TypeInner::Either(ty_l, ty_r) => Some((ty_l, ty_r)),
            _ => None,
        }
    }

    fn as_option(&self) -> Option<&Self> {
        match self.as_inner() {
            TypeInner::Option(ty) => Some(ty),
            _ => None,
        }
    }

    fn is_boolean(&self) -> bool {
        matches!(self.as_inner(), TypeInner::Boolean)
    }

    fn as_integer(&self) -> Option<UIntType> {
        match self.as_inner() {
            TypeInner::UInt(ty) => Some(*ty),
            _ => None,
        }
    }

    fn as_tuple(&self) -> Option<&[Arc<Self>]> {
        match self.as_inner() {
            TypeInner::Tuple(components) => Some(components),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<(&Self, usize)> {
        match self.as_inner() {
            TypeInner::Array(ty, size) => Some((ty, *size)),
            _ => None,
        }
    }

    fn as_list(&self) -> Option<(&Self, NonZeroPow2Usize)> {
        match self.as_inner() {
            TypeInner::List(ty, bound) => Some((ty, *bound)),
            _ => None,
        }
    }
}

impl TreeLike for &ResolvedType {
    fn as_node(&self) -> Tree<Self> {
        match &self.inner {
            TypeInner::Boolean | TypeInner::UInt(..) | TypeInner::Enum(..) | TypeInner::Never => {
                Tree::Nullary
            }
            TypeInner::Option(l) | TypeInner::Array(l, _) | TypeInner::List(l, _) => Tree::Unary(l),
            TypeInner::Either(l, r) => Tree::Binary(l, r),
            TypeInner::Tuple(elements) => Tree::Nary(elements.iter().map(Arc::as_ref).collect()),
        }
    }
}

impl fmt::Debug for ResolvedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Diagnostics are told apart by their debug output, so it is cut off like messages.
        write!(f, "{}", self.in_message())
    }
}

impl fmt::Display for ResolvedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for data in self.verbose_pre_order_iter() {
            data.node.inner.display(f, data.n_children_yielded)?;
        }
        Ok(())
    }
}

impl From<UIntType> for ResolvedType {
    fn from(value: UIntType) -> Self {
        Self::new(TypeInner::UInt(value))
    }
}

/// A type built from a chain of aliases is as deep as the chain,
/// so dropping it recursively could overflow the stack.
/// Instead, the parts that nothing else refers to are moved out and dropped one by one.
impl Drop for ResolvedType {
    fn drop(&mut self) {
        let mut parts = Vec::new();
        self.take_unique_parts(&mut parts);
        while let Some(mut part) = parts.pop() {
            part.take_unique_parts(&mut parts);
        }
    }
}

impl ResolvedType {
    /// Move the parts one level down that nothing else refers to into `parts`,
    /// and leave a leaf in their place.
    fn take_unique_parts(&mut self, parts: &mut Vec<Self>) {
        match std::mem::replace(&mut self.inner, TypeInner::Boolean) {
            TypeInner::Either(left, right) => {
                parts.extend(Arc::into_inner(left));
                parts.extend(Arc::into_inner(right));
            }
            TypeInner::Option(part) | TypeInner::Array(part, _) | TypeInner::List(part, _) => {
                parts.extend(Arc::into_inner(part));
            }
            // The elements can be moved out only if nothing else refers to their slice.
            TypeInner::Tuple(slice) if Arc::strong_count(&slice) == 1 => {
                let elements = slice.to_vec();
                drop(slice);
                parts.extend(elements.into_iter().filter_map(Arc::into_inner));
            }
            TypeInner::Enum(mut info) => info.take_unique_payloads(parts),
            TypeInner::Tuple(_) | TypeInner::Boolean | TypeInner::UInt(_) | TypeInner::Never => {}
        }
    }
}

#[cfg(feature = "arbitrary")]
impl crate::ArbitraryRec for ResolvedType {
    // Deliberately never generates `TypeInner::Never`, which has no values.
    //
    // Deliberately never generates `TypeInner::Enum`.
    // Enum values serialize as bare strings that only resolve against a program's declarations
    // (`UnresolvedValues::resolve`), so the self-contained witness JSON round-trip target (`parse_witness_json_rtt`)
    // would fail by design.
    fn arbitrary_rec(u: &mut arbitrary::Unstructured, budget: usize) -> arbitrary::Result<Self> {
        use arbitrary::Arbitrary;

        match budget.checked_sub(1) {
            None => match u.int_in_range(0..=1)? {
                0 => Ok(Self::boolean()),
                1 => UIntType::arbitrary(u).map(Self::from),
                _ => unreachable!(),
            },
            Some(new_budget) => match u.int_in_range(0..=6)? {
                0 => Ok(Self::boolean()),
                1 => UIntType::arbitrary(u).map(Self::from),
                2 => Self::arbitrary_rec(u, new_budget).map(Self::option),
                3 => {
                    let left = Self::arbitrary_rec(u, new_budget)?;
                    let right = Self::arbitrary_rec(u, new_budget)?;
                    Ok(Self::either(left, right))
                }
                4 => {
                    let len = u.int_in_range(0..=3)?;
                    (0..len)
                        .map(|_| Self::arbitrary_rec(u, new_budget))
                        .collect::<arbitrary::Result<Vec<Self>>>()
                        .map(Self::tuple)
                }
                5 => {
                    let element = Self::arbitrary_rec(u, new_budget)?;
                    let size = u.int_in_range(0..=3)?;
                    Ok(Self::array(element, size))
                }
                6 => {
                    let element = Self::arbitrary_rec(u, new_budget)?;
                    let exp = u.int_in_range(1u32..=4)?;
                    let bound = NonZeroPow2Usize::new_unchecked(2usize.saturating_pow(exp));
                    Ok(Self::list(element, bound))
                }
                _ => unreachable!(),
            },
        }
    }
}

/// ## Panics
///
/// Panics if the type mentions [`TypeInner::Never`].
impl From<&ResolvedType> for StructuralType {
    fn from(value: &ResolvedType) -> Self {
        // Each shared part is lowered once, enum payloads included. A part is popped twice:
        // first to push its parts, then to lower it from them, which are lowered by then.
        let mut done: HashMap<*const ResolvedType, Self> = HashMap::new();
        let mut stack = vec![(value, false)];

        while let Some((ty, parts_done)) = stack.pop() {
            let key = std::ptr::from_ref(ty);
            if done.contains_key(&key) {
                continue;
            }
            if !parts_done {
                stack.push((ty, true));
                ty.for_each_part(|part| stack.push((part, false)));
                continue;
            }

            let lowered = |part: &ResolvedType| done[&std::ptr::from_ref(part)].clone();
            let structural = match ty.as_inner() {
                TypeInner::Either(left, right) => Self::either(lowered(left), lowered(right)),
                TypeInner::Option(inner) => Self::option(lowered(inner)),
                TypeInner::Boolean => Self::boolean(),
                TypeInner::UInt(integer) => Self::from(*integer),
                TypeInner::Tuple(elements) => {
                    Self::tuple(elements.iter().map(Arc::as_ref).map(lowered))
                }
                TypeInner::Array(element, size) => Self::array(lowered(element), *size),
                TypeInner::List(element, bound) => Self::list(lowered(element), *bound),
                TypeInner::Enum(info) => {
                    let payloads = info.variants().iter().map(|v| v.payload_type());
                    Self::balanced_sum(payloads.map(lowered).collect())
                }
                TypeInner::Never => {
                    panic!(
                        "the never type has no structural type; check `is_never` before lowering"
                    )
                }
            };
            done.insert(key, structural);
        }

        let key = std::ptr::from_ref(value);
        done.remove(&key).expect("the type was lowered last")
    }
}

#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for ResolvedType {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        <Self as crate::ArbitraryRec>::arbitrary_rec(u, 3)
    }
}
