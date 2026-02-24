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
