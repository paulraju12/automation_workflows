-- Create interactions table
CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    workflow JSONB,
    timestamp TIMESTAMP NOT NULL
);

-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    interaction_id INTEGER NOT NULL,
    rating VARCHAR(50) NOT NULL,
    comment TEXT,
    FOREIGN KEY (interaction_id) REFERENCES interactions(id)
);