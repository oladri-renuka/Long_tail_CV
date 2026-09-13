"""
SQLite database for human review queue.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np


class ReviewQueueDB:
    """SQLite database for storing flagged images and review metadata."""

    def __init__(self, db_path: str = "app/db/review_queue.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Flagged images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flagged_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    category_id INTEGER,
                    category_name TEXT,
                    predicted_class_id INTEGER,
                    predicted_class_name TEXT,
                    confidence REAL NOT NULL,
                    threshold REAL NOT NULL,
                    routing_reason TEXT,
                    logits_json TEXT,
                    calibrated_probs_json TEXT,
                    flagged_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reviewed BOOLEAN DEFAULT 0,
                    reviewed_timestamp DATETIME,
                    human_label INTEGER,
                    human_label_name TEXT,
                    reviewer_notes TEXT,
                    is_correct BOOLEAN,
                    UNIQUE(image_path)
                )
            """)

            # Review stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_flagged INTEGER,
                    total_reviewed INTEGER,
                    accuracy_on_reviewed REAL,
                    correct_reviews INTEGER,
                    incorrect_reviews INTEGER,
                    queue_size INTEGER
                )
            """)

            # Model metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    head_accuracy REAL,
                    tail_accuracy REAL,
                    ece_before REAL,
                    ece_after REAL,
                    temperature REAL,
                    routing_threshold REAL,
                    tail_routing_rate REAL,
                    head_false_routing_rate REAL,
                    total_routing_rate REAL
                )
            """)

            conn.commit()

    def flag_image(
        self,
        image_path: str,
        category_id: int,
        category_name: str,
        predicted_class_id: int,
        predicted_class_name: str,
        confidence: float,
        threshold: float,
        routing_reason: str,
        logits: Optional[np.ndarray] = None,
        calibrated_probs: Optional[np.ndarray] = None,
    ) -> int:
        """
        Add flagged image to review queue.

        Returns:
            Image ID in database
        """
        logits_json = json.dumps(logits.tolist()) if logits is not None else None
        calibrated_probs_json = json.dumps(calibrated_probs.tolist()) if calibrated_probs is not None else None
        image_name = Path(image_path).name

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO flagged_images (
                        image_path, image_name, category_id, category_name,
                        predicted_class_id, predicted_class_name, confidence,
                        threshold, routing_reason, logits_json, calibrated_probs_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_path, image_name, category_id, category_name,
                    predicted_class_id, predicted_class_name, confidence,
                    threshold, routing_reason, logits_json, calibrated_probs_json
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Image already flagged
                return None

    def mark_reviewed(
        self,
        image_id: int,
        human_label: int,
        human_label_name: str,
        is_correct: bool,
        reviewer_notes: str = "",
    ):
        """Mark image as reviewed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE flagged_images
                SET reviewed = 1, reviewed_timestamp = CURRENT_TIMESTAMP,
                    human_label = ?, human_label_name = ?, is_correct = ?,
                    reviewer_notes = ?
                WHERE id = ?
            """, (human_label, human_label_name, is_correct, reviewer_notes, image_id))
            conn.commit()

    def get_pending_reviews(self, limit: int = 10) -> List[Dict]:
        """Get images pending review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM flagged_images
                WHERE reviewed = 0
                ORDER BY flagged_timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_reviewed_images(self, limit: int = 100) -> List[Dict]:
        """Get reviewed images."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM flagged_images
                WHERE reviewed = 1
                ORDER BY reviewed_timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """Get review queue statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total flagged
            cursor.execute("SELECT COUNT(*) FROM flagged_images")
            total_flagged = cursor.fetchone()[0]

            # Total reviewed
            cursor.execute("SELECT COUNT(*) FROM flagged_images WHERE reviewed = 1")
            total_reviewed = cursor.fetchone()[0]

            # Queue size (pending)
            queue_size = total_flagged - total_reviewed

            # Accuracy on reviewed
            cursor.execute("""
                SELECT COUNT(*), SUM(CAST(is_correct AS INTEGER))
                FROM flagged_images WHERE reviewed = 1
            """)
            result = cursor.fetchone()
            correct_reviews = result[1] if result[1] is not None else 0

            accuracy = correct_reviews / total_reviewed if total_reviewed > 0 else 0.0

            return {
                'total_flagged': total_flagged,
                'total_reviewed': total_reviewed,
                'queue_size': queue_size,
                'correct_reviews': correct_reviews,
                'accuracy': accuracy,
            }

    def log_metrics(
        self,
        head_accuracy: float,
        tail_accuracy: float,
        ece_before: float,
        ece_after: float,
        temperature: float,
        routing_threshold: float,
        tail_routing_rate: float,
        head_false_routing_rate: float,
        total_routing_rate: float,
    ):
        """Log model metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_metrics (
                    head_accuracy, tail_accuracy, ece_before, ece_after,
                    temperature, routing_threshold, tail_routing_rate,
                    head_false_routing_rate, total_routing_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                head_accuracy, tail_accuracy, ece_before, ece_after,
                temperature, routing_threshold, tail_routing_rate,
                head_false_routing_rate, total_routing_rate
            ))
            conn.commit()

    def get_latest_metrics(self) -> Optional[Dict]:
        """Get latest model metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM model_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None

    def clear_database(self):
        """Clear all data (for testing)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM flagged_images")
            cursor.execute("DELETE FROM review_stats")
            cursor.execute("DELETE FROM model_metrics")
            conn.commit()

    def export_csv(self, output_path: str):
        """Export flagged images to CSV."""
        import pandas as pd

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM flagged_images", conn)
            df.to_csv(output_path, index=False)
            print(f"Exported {len(df)} images to {output_path}")

    def get_confusion_matrix_data(self) -> Tuple[List[int], List[int]]:
        """Get true vs predicted labels for reviewed images."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT human_label, predicted_class_id
                FROM flagged_images
                WHERE reviewed = 1 AND human_label IS NOT NULL
            """)
            rows = cursor.fetchall()

        true_labels = [row[0] for row in rows]
        pred_labels = [row[1] for row in rows]

        return true_labels, pred_labels
