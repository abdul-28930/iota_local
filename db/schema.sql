-- New Supabase project: https://aankxlqdhxyerxkharrb.supabase.co
-- Run in SQL Editor. Isolated tables, safe to iterate.

create extension if not exists vector;

create table documents_local (
    id                bigserial primary key,
    document_name     text not null unique,
    source_type       text,
    document_type     text,
    reference_number  text,
    effective_date    date,
    effective_date_hijri text,
    topic_tags        text[],
    total_pages       int,
    created_at        timestamptz default now()
);

create table sama_nora_chunks_local (
    id                bigserial primary key,
    document_id       bigint references documents_local(id) on delete cascade,
    document_name     text not null,
    page_start        int,
    page_end          int,
    section_title     text,
    content           text not null,
    token_count       int,
    language          text,
    dense_embedding   vector(1024),   -- bge-m3 dense dim
    sparse_weights    jsonb,          -- {token_id: weight, ...} from bge-m3 sparse head
    created_at        timestamptz default now()
);

create index idx_local_dense on sama_nora_chunks_local
  using ivfflat (dense_embedding vector_cosine_ops);
create index idx_local_doc on sama_nora_chunks_local (document_id);
create index idx_local_sparse_gin on sama_nora_chunks_local using gin (sparse_weights);

-- Dense-only RPC (sparse scoring done in Python app layer, see retrieve.py)
create or replace function match_chunks_local(query_embedding vector(1024), match_count int)
returns table (
    id bigint, document_name text, page_start int, page_end int,
    section_title text, content text, token_count int, language text,
    similarity float
)
language sql stable as $$
  select id, document_name, page_start, page_end, section_title, content,
         token_count, language,
         1 - (dense_embedding <=> query_embedding) as similarity
  from sama_nora_chunks_local
  order by dense_embedding <=> query_embedding
  limit match_count;
$$;

-- Multi-reference metadata support (dates/numbers can appear multiple times, different meanings)
alter table documents_local add column if not exists extracted_references jsonb;

-- Feedback on generated answers
create table if not exists feedback_local (
    id bigserial primary key,
    question text not null,
    answer text not null,
    rating smallint not null check (rating in (0, 1)),  -- 0 = down, 1 = up
    created_at timestamptz default now()
);

-- Conversation sessions + messages, for history/session switching
create table if not exists sessions_local (
    id uuid primary key default gen_random_uuid(),
    title text,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table if not exists messages_local (
    id bigserial primary key,
    session_id uuid references sessions_local(id) on delete cascade,
    role text not null check (role in ('user','bot')),
    content text not null,
    sources jsonb,
    created_at timestamptz default now()
);

create index if not exists idx_messages_session on messages_local(session_id, created_at);

-- Auto-touch session.updated_at whenever a new message is inserted
create or replace function touch_session_updated_at() returns trigger as $$
begin
  update sessions_local set updated_at = now() where id = new.session_id;
  return new;
end;
$$ language plpgsql;

create trigger trg_touch_session
after insert on messages_local
for each row execute function touch_session_updated_at();

-- Scope sessions to a user (fingerprint ID), needed for correct multi-browser/user isolation
alter table sessions_local add column if not exists user_id text;
create index if not exists idx_sessions_user on sessions_local(user_id);

-- Long-term (cross-session) memory: extracted facts/preferences per user,
-- injected into every session's prompt regardless of which chat it is.
create table if not exists user_memory_local (
    id bigserial primary key,
    user_id text not null,
    fact text not null,
    source_session_id uuid references sessions_local(id) on delete set null,
    created_at timestamptz default now()
);
create index if not exists idx_user_memory_user on user_memory_local(user_id);

-- Content-hash dedup: prevents the same underlying PDF from being ingested
-- multiple times under different filenames (e.g. UUID-prefixed vs plain name)
alter table documents_local add column if not exists content_hash text;
create index if not exists idx_documents_content_hash on documents_local(content_hash);
