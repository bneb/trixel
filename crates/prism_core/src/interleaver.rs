//! 2D Spatial Interleaver for distributing LDPC codeword bits across the image canvas.
//!
//! Camera channels suffer from spatial burst errors (local glare, smudges, finger occlusions).
//! This interleaver maps consecutive 1D codeword bits into widely separated 2D coordinates
//! using a coprime coordinate permutation, maximizing the coding gain of Belief Propagation.

#[derive(Clone, Debug)]
pub struct SpatialInterleaver {
    pub width: usize,
    pub height: usize,
    coords: Vec<(usize, usize)>,
}

impl SpatialInterleaver {
    pub fn new(width: usize, height: usize) -> Self {
        let total = width * height;
        assert!(total > 0, "Grid dimensions must be positive");

        let mut coords = Vec::with_capacity(total);
        let mut visited = vec![false; total];

        // Diagonal pseudo-random scan
        let p_x = find_coprime(width, 13);
        let p_y = find_coprime(height, 17);

        for i in 0..total {
            let x = (i * p_x) % width;
            let y = (i * p_y + (i / width)) % height;
            let idx = y * width + x;
            if !visited[idx] {
                visited[idx] = true;
                coords.push((x, y));
            }
        }

        // Fill any remaining cells sequentially
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if !visited[idx] {
                    visited[idx] = true;
                    coords.push((x, y));
                }
            }
        }

        Self {
            width,
            height,
            coords,
        }
    }

    #[inline]
    pub fn index_to_coord(&self, bit_index: usize) -> (usize, usize) {
        self.coords[bit_index % self.coords.len()]
    }

    pub fn interleave<T: Clone>(&self, input: &[T]) -> Vec<T> {
        assert_eq!(input.len(), self.coords.len());
        let mut out = input.to_vec();
        for (i, &c) in self.coords.iter().enumerate() {
            let flat_idx = c.1 * self.width + c.0;
            out[flat_idx] = input[i].clone();
        }
        out
    }

    pub fn deinterleave<T: Clone>(&self, input: &[T]) -> Vec<T> {
        assert_eq!(input.len(), self.coords.len());
        let mut out = input.to_vec();
        for (i, &c) in self.coords.iter().enumerate() {
            let flat_idx = c.1 * self.width + c.0;
            out[i] = input[flat_idx].clone();
        }
        out
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn find_coprime(n: usize, seed: usize) -> usize {
    let mut candidate = seed.max(1);
    while gcd(candidate, n) != 1 {
        candidate += 1;
    }
    candidate
}
