"""Initial schema — users, conversations, messages, memories with pgvector.

Revision ID: 001
Revises: None
Create Date: 2026-02-17
"""
from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.execute("""
        CREATE TABLE users (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            phone_number VARCHAR(20) UNIQUE NOT NULL,
            display_name VARCHAR(255),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'
        )
    """)

    op.execute("""
        CREATE TABLE conversations (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            channel VARCHAR(10) NOT NULL CHECK (channel IN ('voice', 'text')),
            model_used VARCHAR(50),
            started_at TIMESTAMPTZ DEFAULT NOW(),
            ended_at TIMESTAMPTZ,
            summary TEXT,
            metadata JSONB DEFAULT '{}'
        )
    """)

    op.execute("""
        CREATE TABLE messages (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'
        )
    """)

    op.execute("""
        CREATE TABLE memories (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            type VARCHAR(20) NOT NULL CHECK (type IN ('fact', 'summary', 'preference', 'action_item')),
            content TEXT NOT NULL,
            embedding vector(1536),
            confidence FLOAT DEFAULT 1.0,
            source_channel VARCHAR(10) CHECK (source_channel IN ('voice', 'text')),
            source_conversation_id UUID REFERENCES conversations(id),
            supersedes_id UUID REFERENCES memories(id),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        )
    """)

    op.execute("CREATE INDEX idx_users_phone ON users(phone_number)")
    op.execute("CREATE INDEX idx_conversations_user_id ON conversations(user_id)")
    op.execute("CREATE INDEX idx_messages_conversation_id ON messages(conversation_id)")
    op.execute("CREATE INDEX idx_memories_user_id ON memories(user_id)")
    op.execute(
        "CREATE INDEX idx_memories_embedding ON memories "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS memories CASCADE")
    op.execute("DROP TABLE IF EXISTS messages CASCADE")
    op.execute("DROP TABLE IF EXISTS conversations CASCADE")
    op.execute("DROP TABLE IF EXISTS users CASCADE")
