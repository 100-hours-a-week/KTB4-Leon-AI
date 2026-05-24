from fastapi import APIRouter, HTTPException

from app.controllers import posts_controller
from app.schemas import Post, PostCreate

router = APIRouter(prefix="/posts", tags=["posts"])


@router.get("/", response_model=list[Post])
def read_posts():
    return posts_controller.get_posts()


@router.get("/{post_id}", response_model=Post)
def read_post(post_id: int):
    post = posts_controller.get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


@router.post("/", response_model=Post)
def create_post(post: PostCreate):
    return posts_controller.create_post(post)


@router.delete("/{post_id}")
def delete_post(post_id: int):
    deleted = posts_controller.delete_post(post_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"message": "Post deleted"}
