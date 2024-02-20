CREATE TABLE IF NOT EXISTS events (
  active INTEGER DEFAULT 1, 
  secret_code TEXT PRIMARY KEY,
  user_id TEXT NOT NULL, 
  name TEXT NOT NULL,
  details TEXT, 
  start_datetime INTEGER NOT NULL,
  end_datetime INTEGER,
  INDEX active_index (active),
  INDEX start_end_index (start_datetime, end_datetime)
);

CREATE TABLE IF NOT EXISTS people (
  user_id TEXT PRIMARY KEY, 
  display_name TEXT NOT NULL,
  email TEXT NOT NULL, 
  salt TEXT NOT NULL,
  password TEXT NOT NULL, 
  cell_phone TEXT NOT NULL,
  INDEX email_index (email)
);

CREATE TABLE IF NOT EXISTS rsvp (
  rsvp_id TEXT PRIMARY KEY, 
  user_id TEXT NOT NULL,
  event_id TEXT NOT NULL, 
  rsvp_status TEXT NOT NULL,
  INDEX user_event_index (user_id, event_id)
);

CREATE TABLE IF NOT EXISTS config (
  KEY TEXT PRIMARY KEY, 
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, 
  user_id TEXT NOT NULL,
  expire_time INTEGER NOT NULL,
  INDEX expire_index (expire_time)
);