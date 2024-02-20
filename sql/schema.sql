CREATE TABLE IF NOT EXISTS events (
  active INTEGER DEFAULT 1,
  secret_code TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  details TEXT,
  start_datetime INTEGER NOT NULL,
  end_datetime INTEGER
);

CREATE INDEX IF NOT EXISTS active_index ON events(active);
CREATE INDEX IF NOT EXISTS start_end_index ON events(start_datetime, end_datetime);

CREATE TABLE IF NOT EXISTS people (
  user_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  email TEXT NOT NULL,
  salt TEXT NOT NULL,
  password TEXT NOT NULL,
  cell_phone TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS email_index ON people(email);

CREATE TABLE IF NOT EXISTS rsvp (
  rsvp_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  rsvp_status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS user_event_index ON rsvp(user_id, event_id);

CREATE TABLE IF NOT EXISTS config (
  KEY TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  expire_time INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS expire_index ON sessions(expire_time);