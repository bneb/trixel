//! # Geometry Utilities
//!
//! Douglas-Peucker polygon simplification and L-bracket shape classification
//! for the computer vision pipeline.

/// A 2D point (using f64 for sub-pixel precision during perspective correction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

/// Douglas-Peucker polygon simplification.
///
/// Recursively reduces a polyline to fewer vertices while keeping the shape
/// within `epsilon` perpendicular distance of the original.
pub fn douglas_peucker(points: &[Point], epsilon: f64) -> Vec<Point> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    // Find the point furthest from the line between first and last
    let (first, last) = (points[0], points[points.len() - 1]);
    let mut max_dist = 0.0;
    let mut max_idx = 0;

    for (i, p) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let d = perpendicular_distance(*p, first, last);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        // Recursively simplify both halves
        let mut left = douglas_peucker(&points[..=max_idx], epsilon);
        let right = douglas_peucker(&points[max_idx..], epsilon);
        left.pop(); // Remove duplicate split point
        left.extend_from_slice(&right);
        left
    } else {
        // All intermediate points are within epsilon — keep only endpoints
        vec![first, last]
    }
}

/// Perpendicular distance from point `p` to the line through `a` and `b`.
fn perpendicular_distance(p: Point, a: Point, b: Point) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let line_len_sq = dx * dx + dy * dy;

    if line_len_sq < 1e-12 {
        // a and b are the same point
        let ex = p.x - a.x;
        let ey = p.y - a.y;
        return (ex * ex + ey * ey).sqrt();
    }

    let numerator = ((dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) as f64).abs();
    numerator / line_len_sq.sqrt()
}

/// Check if a polygon with exactly 6 vertices forms an L-bracket shape.
///
/// An L-bracket has:
/// - Exactly 6 vertices
/// - All angles are approximately 90° or 270°
/// - The shape has a concave notch (one reflex angle)
pub fn is_l_shape(polygon: &[Point]) -> bool {
    if polygon.len() != 6 {
        return false;
    }

    // Check that all edges are approximately axis-aligned
    // (angles between consecutive edges should be ~90°)
    let mut right_angle_count = 0;
    let n = polygon.len();

    for i in 0..n {
        let p0 = polygon[i];
        let p1 = polygon[(i + 1) % n];
        let p2 = polygon[(i + 2) % n];

        let dx1 = p1.x - p0.x;
        let dy1 = p1.y - p0.y;
        let dx2 = p2.x - p1.x;
        let dy2 = p2.y - p1.y;

        // Dot product should be ~0 for perpendicular edges
        let dot = dx1 * dx2 + dy1 * dy2;
        let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
        let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();

        if len1 < 1e-6 || len2 < 1e-6 {
            return false;
        }

        let cos_angle = dot / (len1 * len2);
        // Allow ±15° tolerance from 90°
        if cos_angle.abs() < 0.26 {
            // cos(75°) ≈ 0.26, so |cos| < 0.26 means angle is within 75°-105°
            right_angle_count += 1;
        }
    }

    // An L-shape should have 6 approximately-right angles
    right_angle_count >= 5
}

/// Classify 4 detected L-brackets into TL, TR, BL, BR based on their
/// centroid positions.
///
/// Returns `[TL, TR, BL, BR]` indices into the input array.
/// Returns `None` if there aren't exactly 4 brackets.
pub fn classify_corners(centroids: &[Point]) -> Option<[usize; 4]> {
    if centroids.len() != 4 {
        return None;
    }

    // Find the center of all centroids
    let cx: f64 = centroids.iter().map(|p| p.x).sum::<f64>() / 4.0;
    let cy: f64 = centroids.iter().map(|p| p.y).sum::<f64>() / 4.0;

    let mut tl = None;
    let mut tr = None;
    let mut bl = None;
    let mut br = None;

    for (i, p) in centroids.iter().enumerate() {
        let is_left = p.x < cx;
        let is_top = p.y < cy;
        match (is_left, is_top) {
            (true, true) => tl = Some(i),
            (false, true) => tr = Some(i),
            (true, false) => bl = Some(i),
            (false, false) => br = Some(i),
        }
    }

    Some([tl?, tr?, bl?, br?])
}

/// Compute the centroid of a polygon.
pub fn centroid(polygon: &[Point]) -> Point {
    let n = polygon.len() as f64;
    Point {
        x: polygon.iter().map(|p| p.x).sum::<f64>() / n,
        y: polygon.iter().map(|p| p.y).sum::<f64>() / n,
    }
}

// ---------------------------------------------------------------------------
// Triangle-Specific Geometry
// ---------------------------------------------------------------------------

/// Signed area of triangle `(a, b, c)` via the shoelace formula.
/// Positive = counter-clockwise, negative = clockwise.
pub fn triangle_area_signed(a: Point, b: Point, c: Point) -> f64 {
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
}

/// Absolute area of triangle `(a, b, c)`.
pub fn triangle_area(a: Point, b: Point, c: Point) -> f64 {
    triangle_area_signed(a, b, c).abs()
}

/// Check if a 3-vertex polygon forms a valid (non-degenerate) triangle.
///
/// Rejects triangles that are too small (noise) or too large (full-image).
/// Also rejects near-collinear triangles (area < `min_area`).
pub fn is_valid_triangle(polygon: &[Point], min_area: f64, max_area: f64) -> bool {
    if polygon.len() != 3 {
        return false;
    }
    let area = triangle_area(polygon[0], polygon[1], polygon[2]);
    area >= min_area && area <= max_area
}

/// Classify 3 detected anchor centroids into (TL, TR, BL).
///
/// Strategy: the triangle formed by the 3 anchors has:
/// - TL = top-left (minimum x+y sum = closest to origin)
/// - TR = top-right (maximum x - y)
/// - BL = bottom-left (minimum x - y)
///
/// Returns indices `[tl_idx, tr_idx, bl_idx]` into the input array.
/// Returns `None` if fewer than 3 centroids.
pub fn classify_tri_corners(centroids: &[Point]) -> Option<[usize; 3]> {
    if centroids.len() < 3 {
        return None;
    }

    // TL: smallest x + y (closest to top-left corner)
    let tl = (0..3)
        .min_by(|&a, &b| {
            let sa = centroids[a].x + centroids[a].y;
            let sb = centroids[b].x + centroids[b].y;
            sa.partial_cmp(&sb).unwrap()
        })
        .unwrap();

    // TR: largest x - y (far right, near top)
    let tr = (0..3)
        .filter(|&i| i != tl)
        .max_by(|&a, &b| {
            let da = centroids[a].x - centroids[a].y;
            let db = centroids[b].x - centroids[b].y;
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();

    // BL: the remaining one
    let bl = (0..3).find(|&i| i != tl && i != tr).unwrap();

    Some([tl, tr, bl])
}

/// Compute a 2×3 affine transformation matrix that maps `src` → `dst`.
///
/// Given three source points and three destination points, solves the system:
/// ```text
///   dst_x = a₀₀·src_x + a₀₁·src_y + a₀₂
///   dst_y = a₁₀·src_x + a₁₁·src_y + a₁₂
/// ```
///
/// Returns `[[a₀₀, a₀₁, a₀₂], [a₁₀, a₁₁, a₁₂]]` or `None` if degenerate.
pub fn affine_from_triangles(src: [Point; 3], dst: [Point; 3]) -> Option<[[f64; 3]; 2]> {
    // Solve the 3×3 system for x-coordinates:
    // | src[0].x  src[0].y  1 |   | a₀₀ |   | dst[0].x |
    // | src[1].x  src[1].y  1 | × | a₀₁ | = | dst[1].x |
    // | src[2].x  src[2].y  1 |   | a₀₂ |   | dst[2].x |

    let det = src[0].x * (src[1].y - src[2].y)
            - src[0].y * (src[1].x - src[2].x)
            + (src[1].x * src[2].y - src[2].x * src[1].y);

    if det.abs() < 1e-10 {
        return None; // Degenerate (collinear source points)
    }

    let inv_det = 1.0 / det;

    // Cofactor matrix (transposed = inverse for 3×3)
    let c00 = src[1].y - src[2].y;
    let c01 = src[2].y - src[0].y;
    let c02 = src[0].y - src[1].y;
    let c10 = src[2].x - src[1].x;
    let c11 = src[0].x - src[2].x;
    let c12 = src[1].x - src[0].x;
    let c20 = src[1].x * src[2].y - src[2].x * src[1].y;
    let c21 = src[2].x * src[0].y - src[0].x * src[2].y;
    let c22 = src[0].x * src[1].y - src[1].x * src[0].y;

    // Solve for x-row
    let a00 = inv_det * (c00 * dst[0].x + c01 * dst[1].x + c02 * dst[2].x);
    let a01 = inv_det * (c10 * dst[0].x + c11 * dst[1].x + c12 * dst[2].x);
    let a02 = inv_det * (c20 * dst[0].x + c21 * dst[1].x + c22 * dst[2].x);

    // Solve for y-row
    let a10 = inv_det * (c00 * dst[0].y + c01 * dst[1].y + c02 * dst[2].y);
    let a11 = inv_det * (c10 * dst[0].y + c11 * dst[1].y + c12 * dst[2].y);
    let a12 = inv_det * (c20 * dst[0].y + c21 * dst[1].y + c22 * dst[2].y);

    Some([[a00, a01, a02], [a10, a11, a12]])
}

/// Apply an affine transformation to a point.
pub fn affine_transform(m: &[[f64; 3]; 2], p: Point) -> Point {
    Point {
        x: m[0][0] * p.x + m[0][1] * p.y + m[0][2],
        y: m[1][0] * p.x + m[1][1] * p.y + m[1][2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_area_unit_triangle() {
        // Right triangle with vertices at (0,0), (1,0), (0,1) → area = 0.5
        let a = triangle_area(Point::new(0.0, 0.0), Point::new(1.0, 0.0), Point::new(0.0, 1.0));
        assert!((a - 0.5).abs() < 1e-10);
    }

    #[test]
    fn triangle_area_large() {
        let a = triangle_area(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(50.0, 100.0),
        );
        assert!((a - 5000.0).abs() < 1e-6);
    }

    #[test]
    fn triangle_area_collinear_is_zero() {
        let a = triangle_area(Point::new(0.0, 0.0), Point::new(5.0, 5.0), Point::new(10.0, 10.0));
        assert!(a < 1e-10, "Collinear points → zero area");
    }

    #[test]
    fn is_valid_triangle_rejects_degenerate() {
        let pts = vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0), Point::new(2.0, 2.0)];
        assert!(!is_valid_triangle(&pts, 1.0, 10000.0));
    }

    #[test]
    fn is_valid_triangle_accepts_good() {
        let pts = vec![Point::new(0.0, 0.0), Point::new(10.0, 0.0), Point::new(5.0, 8.0)];
        assert!(is_valid_triangle(&pts, 1.0, 10000.0));
    }

    #[test]
    fn classify_tri_corners_basic() {
        let centroids = vec![
            Point::new(10.0, 10.0),   // TL
            Point::new(200.0, 15.0),  // TR
            Point::new(12.0, 200.0),  // BL
        ];
        let [tl, tr, bl] = classify_tri_corners(&centroids).unwrap();
        assert_eq!(tl, 0);
        assert_eq!(tr, 1);
        assert_eq!(bl, 2);
    }

    #[test]
    fn classify_tri_corners_shuffled() {
        let centroids = vec![
            Point::new(12.0, 200.0),  // BL (idx 0)
            Point::new(10.0, 10.0),   // TL (idx 1)
            Point::new(200.0, 15.0),  // TR (idx 2)
        ];
        let [tl, tr, bl] = classify_tri_corners(&centroids).unwrap();
        assert_eq!(tl, 1);
        assert_eq!(tr, 2);
        assert_eq!(bl, 0);
    }

    #[test]
    fn affine_identity_from_same_triangles() {
        let tri = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(0.0, 100.0),
        ];
        let m = affine_from_triangles(tri, tri).unwrap();

        // Should be identity: [[1,0,0],[0,1,0]]
        assert!((m[0][0] - 1.0).abs() < 1e-10);
        assert!((m[0][1]).abs() < 1e-10);
        assert!((m[0][2]).abs() < 1e-10);
        assert!((m[1][0]).abs() < 1e-10);
        assert!((m[1][1] - 1.0).abs() < 1e-10);
        assert!((m[1][2]).abs() < 1e-10);
    }

    #[test]
    fn affine_maps_corners_correctly() {
        let src = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(0.0, 100.0),
        ];
        let dst = [
            Point::new(10.0, 20.0),
            Point::new(110.0, 20.0),
            Point::new(10.0, 120.0),
        ];
        let m = affine_from_triangles(src, dst).unwrap();

        for i in 0..3 {
            let mapped = affine_transform(&m, src[i]);
            assert!((mapped.x - dst[i].x).abs() < 1e-6, "x mismatch at {i}");
            assert!((mapped.y - dst[i].y).abs() < 1e-6, "y mismatch at {i}");
        }
    }

    #[test]
    fn affine_degenerate_returns_none() {
        let src = [
            Point::new(0.0, 0.0),
            Point::new(5.0, 5.0),
            Point::new(10.0, 10.0),  // collinear
        ];
        let dst = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
        ];
        assert!(affine_from_triangles(src, dst).is_none());
    }
}

