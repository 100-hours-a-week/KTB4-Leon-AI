from app.database import get_connection
from app.schemas import PostCreate


def get_posts():
    conn = get_connection()
    rows = conn.execute("SELECT id, title, content FROM posts ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_post(post_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT id, title, content FROM posts WHERE id = ?",
        (post_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def create_post(post: PostCreate):
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO posts (title, content) VALUES (?, ?)",
        (post.title, post.content),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return get_post(new_id)


def delete_post(post_id: int):
    conn = get_connection()
    cursor = conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    conn.commit()
    deleted_count = cursor.rowcount
    conn.close()
    return deleted_count > 0
