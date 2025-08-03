-- Initial database schema for CoT SafePath Filter
-- Migration: 001_initial_schema.sql

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Filter operations table (main audit log)
CREATE TABLE filter_operations (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64) UNIQUE NOT NULL,
    session_id VARCHAR(64),
    user_id VARCHAR(64),
    input_hash VARCHAR(64) NOT NULL,
    content_length INTEGER NOT NULL,
    safety_score DECIMAL(3,2) NOT NULL CHECK (safety_score >= 0 AND safety_score <= 1),
    was_filtered BOOLEAN NOT NULL DEFAULT FALSE,
    filter_reasons TEXT[],
    processing_time_ms INTEGER NOT NULL CHECK (processing_time_ms >= 0),
    cache_hit BOOLEAN DEFAULT FALSE,
    safety_level VARCHAR(20) NOT NULL,
    filter_threshold DECIMAL(3,2) NOT NULL,
    metadata JSONB DEFAULT '{}',
    context TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Safety detections table
CREATE TABLE safety_detections (
    id SERIAL PRIMARY KEY,
    operation_id INTEGER NOT NULL REFERENCES filter_operations(id) ON DELETE CASCADE,
    detector_name VARCHAR(50) NOT NULL,
    detector_version VARCHAR(20) DEFAULT '1.0',
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    detected_patterns TEXT[],
    severity VARCHAR(20) NOT NULL,
    is_harmful BOOLEAN NOT NULL,
    reasoning TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Filter rules table
CREATE TABLE filter_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    pattern TEXT,
    pattern_type VARCHAR(20) DEFAULT 'regex',
    action VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    threshold DECIMAL(3,2) DEFAULT 0.7 CHECK (threshold >= 0 AND threshold <= 1),
    enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 0,
    category VARCHAR(50),
    tags TEXT[],
    usage_count INTEGER DEFAULT 0,
    last_triggered TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    role VARCHAR(20) DEFAULT 'user',
    permissions TEXT[],
    api_key VARCHAR(64) UNIQUE,
    api_key_active BOOLEAN DEFAULT FALSE,
    rate_limit_override INTEGER,
    request_count INTEGER DEFAULT 0,
    last_login TIMESTAMP WITH TIME ZONE,
    last_request TIMESTAMP WITH TIME ZONE,
    full_name VARCHAR(100),
    organization VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Sessions table
CREATE TABLE sessions (
    id VARCHAR(64) PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    data JSONB DEFAULT '{}',
    ip_address INET,
    user_agent VARCHAR(500),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- System metrics table
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(50) NOT NULL,
    metric_type VARCHAR(20) NOT NULL,
    value DECIMAL(15,6) NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    aggregation_window VARCHAR(20)
);

-- Audit logs table
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    event_category VARCHAR(30) NOT NULL,
    user_id UUID,
    session_id VARCHAR(64),
    ip_address INET,
    resource_type VARCHAR(50),
    resource_id VARCHAR(64),
    action VARCHAR(50) NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    details JSONB DEFAULT '{}',
    old_values JSONB,
    new_values JSONB,
    risk_level VARCHAR(20) DEFAULT 'low',
    requires_review BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create indexes for performance
-- Filter operations indexes
CREATE INDEX idx_filter_ops_request_id ON filter_operations(request_id);
CREATE INDEX idx_filter_ops_session_id ON filter_operations(session_id);
CREATE INDEX idx_filter_ops_user_id ON filter_operations(user_id);
CREATE INDEX idx_filter_ops_input_hash ON filter_operations(input_hash);
CREATE INDEX idx_filter_ops_created_at ON filter_operations(created_at);
CREATE INDEX idx_filter_ops_safety_score ON filter_operations(safety_score);
CREATE INDEX idx_filter_ops_filtered ON filter_operations(was_filtered);
CREATE INDEX idx_filter_ops_user_created ON filter_operations(user_id, created_at);

-- Safety detections indexes
CREATE INDEX idx_detections_operation_id ON safety_detections(operation_id);
CREATE INDEX idx_detections_detector_name ON safety_detections(detector_name);
CREATE INDEX idx_detections_confidence ON safety_detections(confidence);
CREATE INDEX idx_detections_harmful ON safety_detections(is_harmful);
CREATE INDEX idx_detections_operation_detector ON safety_detections(operation_id, detector_name);

-- Filter rules indexes
CREATE INDEX idx_rules_enabled ON filter_rules(enabled);
CREATE INDEX idx_rules_category ON filter_rules(category);
CREATE INDEX idx_rules_priority ON filter_rules(priority);
CREATE INDEX idx_rules_updated_at ON filter_rules(updated_at);

-- Users indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_api_key ON users(api_key);
CREATE INDEX idx_users_last_login ON users(last_login);
CREATE INDEX idx_users_active ON users(is_active);

-- Sessions indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX idx_sessions_active ON sessions(is_active);
CREATE INDEX idx_sessions_last_accessed ON sessions(last_accessed);

-- System metrics indexes
CREATE INDEX idx_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
CREATE INDEX idx_metrics_type ON system_metrics(metric_type);
CREATE INDEX idx_metrics_timestamp ON system_metrics(timestamp);
CREATE UNIQUE INDEX uq_metric_time_window ON system_metrics(metric_name, timestamp, aggregation_window);

-- Audit logs indexes
CREATE INDEX idx_audit_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_user_timestamp ON audit_logs(user_id, timestamp);
CREATE INDEX idx_audit_category ON audit_logs(event_category);
CREATE INDEX idx_audit_risk_level ON audit_logs(risk_level);
CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_requires_review ON audit_logs(requires_review);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_filter_rules_updated_at 
    BEFORE UPDATE ON filter_rules 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default filter rules
INSERT INTO filter_rules (name, description, pattern, pattern_type, action, severity, category, priority) VALUES
('blocked_tokens', 'Block dangerous tokens', '(kill|murder|bomb|explosive|poison|weapon)', 'regex', 'block', 'high', 'content_filter', 100),
('harmful_planning', 'Detect harmful planning sequences', 'step \d+.*(acquire|obtain).*(dangerous|harmful|illegal)', 'regex', 'flag', 'high', 'planning_filter', 90),
('manipulation_patterns', 'Detect manipulation techniques', '(gradually|step by step).*(manipulate|deceive|trick)', 'regex', 'flag', 'medium', 'behavioral_filter', 80),
('trust_exploitation', 'Detect trust-then-exploit patterns', '(gain|establish).*(trust|confidence).*(exploit|use)', 'regex', 'flag', 'high', 'behavioral_filter', 85);

-- Insert default admin user (password: admin123 - change in production!)
INSERT INTO users (username, email, password_hash, role, is_verified, permissions) VALUES
('admin', 'admin@safepath.local', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj.8TQrF15Zu', 'admin', true, ARRAY['admin', 'manage_users', 'manage_rules', 'view_all_operations']);

COMMIT;