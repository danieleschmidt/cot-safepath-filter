-- Database initialization script for CoT SafePath Filter
-- Creates necessary tables, indexes, and initial data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS safepath;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO safepath, public;

-- Filter operations audit log
CREATE TABLE IF NOT EXISTS filter_operations (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    input_hash VARCHAR(64) NOT NULL,
    input_size INTEGER,
    safety_score DECIMAL(3,2),
    filtered BOOLEAN NOT NULL,
    filter_reason TEXT,
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    filter_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Safety detection results
CREATE TABLE IF NOT EXISTS safety_detections (
    id SERIAL PRIMARY KEY,
    operation_id INTEGER REFERENCES filter_operations(id) ON DELETE CASCADE,
    detector_name VARCHAR(50) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    detected_patterns TEXT[],
    severity VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Filter rules configuration
CREATE TABLE IF NOT EXISTS filter_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    pattern TEXT,
    action VARCHAR(20) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    category VARCHAR(50),
    enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User feedback for adaptive learning
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    operation_id INTEGER REFERENCES filter_operations(id) ON DELETE CASCADE,
    feedback_type VARCHAR(20) NOT NULL, -- 'false_positive', 'false_negative', 'correct'
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    comments TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,3) NOT NULL,
    metric_unit VARCHAR(20),
    timestamp TIMESTAMP DEFAULT NOW(),
    labels JSONB
);

-- System configuration
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    config_type VARCHAR(20) DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- API keys and authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    key_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    permissions JSONB DEFAULT '{}',
    rate_limit INTEGER DEFAULT 1000,
    enabled BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP
);

-- Content cache for performance
CREATE TABLE IF NOT EXISTS content_cache (
    id SERIAL PRIMARY KEY,
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    cached_result JSONB NOT NULL,
    cache_version VARCHAR(20) NOT NULL,
    hits INTEGER DEFAULT 0,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit schema tables
SET search_path TO audit, public;

-- Audit log for all system changes
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(10) NOT NULL, -- INSERT, UPDATE, DELETE
    old_values JSONB,
    new_values JSONB,
    user_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW(),
    client_ip INET,
    user_agent TEXT
);

-- Security events log
CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    description TEXT NOT NULL,
    source_ip INET,
    user_agent TEXT,
    additional_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Monitoring schema tables
SET search_path TO monitoring, public;

-- System health metrics
CREATE TABLE IF NOT EXISTS health_checks (
    id SERIAL PRIMARY KEY,
    check_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'healthy', 'degraded', 'unhealthy'
    response_time_ms INTEGER,
    error_message TEXT,
    checked_at TIMESTAMP DEFAULT NOW()
);

-- Resource usage metrics
CREATE TABLE IF NOT EXISTS resource_usage (
    id SERIAL PRIMARY KEY,
    cpu_percent DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    disk_percent DECIMAL(5,2),
    network_in_bytes BIGINT,
    network_out_bytes BIGINT,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Reset search path
SET search_path TO safepath, public;

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_operations_created_at ON filter_operations(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_operations_request_id ON filter_operations(request_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_operations_user_id ON filter_operations(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_operations_safety_score ON filter_operations(safety_score);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_operations_filtered ON filter_operations(filtered);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_detections_operation_id ON safety_detections(operation_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_detections_detector_name ON safety_detections(detector_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_safety_detections_confidence ON safety_detections(confidence);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_rules_enabled ON filter_rules(enabled);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_rules_category ON filter_rules(category);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_filter_rules_priority ON filter_rules(priority);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_name_timestamp ON performance_metrics(metric_name, timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_labels ON performance_metrics USING GIN(labels);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_content_cache_hash ON content_cache(content_hash);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_content_cache_expires ON content_cache(expires_at);

-- Audit schema indexes
SET search_path TO audit, public;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_table_timestamp ON audit_log(table_name, timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_type_timestamp ON security_events(event_type, created_at);

-- Monitoring schema indexes
SET search_path TO monitoring, public;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_health_checks_name_timestamp ON health_checks(check_name, checked_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_resource_usage_timestamp ON resource_usage(recorded_at);

-- Reset search path
SET search_path TO safepath, public;

-- Insert default configuration
INSERT INTO system_config (config_key, config_value, config_type, description) VALUES
    ('default_safety_level', 'balanced', 'string', 'Default safety level for filtering'),
    ('max_request_size', '10485760', 'integer', 'Maximum request size in bytes (10MB)'),
    ('cache_ttl', '3600', 'integer', 'Cache TTL in seconds'),
    ('rate_limit_default', '1000', 'integer', 'Default rate limit per hour'),
    ('enable_adaptive_filtering', 'false', 'boolean', 'Enable adaptive filtering based on feedback'),
    ('log_filtered_content', 'false', 'boolean', 'Whether to log filtered content for review'),
    ('metrics_retention_days', '30', 'integer', 'Number of days to retain metrics'),
    ('audit_retention_days', '90', 'integer', 'Number of days to retain audit logs')
ON CONFLICT (config_key) DO NOTHING;

-- Insert default filter rules
INSERT INTO filter_rules (name, description, pattern, action, severity, category, priority) VALUES
    ('harmful_instructions', 'Detect harmful instructions', 'how to (make|create|build) .* (weapon|bomb|explosive)', 'block', 'high', 'violence', 100),
    ('bypass_attempts', 'Detect security bypass attempts', '(bypass|circumvent|evade) .* (security|detection|filter)', 'block', 'high', 'security', 90),
    ('deceptive_patterns', 'Detect deceptive reasoning patterns', '(hide|conceal|mask) .* (intent|purpose|goal)', 'flag', 'medium', 'deception', 80),
    ('data_exfiltration', 'Detect data exfiltration attempts', '(extract|steal|copy) .* (data|information|credentials)', 'block', 'high', 'security', 95),
    ('social_engineering', 'Detect social engineering attempts', '(manipulate|trick|deceive) .* (user|person|target)', 'flag', 'medium', 'manipulation', 70)
ON CONFLICT (name) DO NOTHING;

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_filter_operations_updated_at BEFORE UPDATE ON filter_operations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_filter_rules_updated_at BEFORE UPDATE ON filter_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for audit logging
CREATE OR REPLACE FUNCTION audit.log_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit.audit_log (table_name, operation, old_values, new_values, user_id, client_ip)
    VALUES (
        TG_TABLE_NAME,
        TG_OP,
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END,
        current_setting('app.current_user_id', true),
        inet_client_addr()
    );
    
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for important tables
CREATE TRIGGER filter_rules_audit AFTER INSERT OR UPDATE OR DELETE ON filter_rules FOR EACH ROW EXECUTE FUNCTION audit.log_changes();
CREATE TRIGGER system_config_audit AFTER INSERT OR UPDATE OR DELETE ON system_config FOR EACH ROW EXECUTE FUNCTION audit.log_changes();
CREATE TRIGGER api_keys_audit AFTER INSERT OR UPDATE OR DELETE ON api_keys FOR EACH ROW EXECUTE FUNCTION audit.log_changes();

-- Create materialized view for performance dashboards
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_filter_stats AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN filtered THEN 1 END) as filtered_requests,
    AVG(safety_score) as avg_safety_score,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(DISTINCT user_id) as unique_users
FROM filter_operations
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_filter_stats_date ON daily_filter_stats(date);

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_dashboard_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_filter_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA safepath TO PUBLIC;
GRANT USAGE ON SCHEMA audit TO PUBLIC;
GRANT USAGE ON SCHEMA monitoring TO PUBLIC;

-- Grant table permissions (adjust as needed for security)
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA safepath TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO PUBLIC;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA monitoring TO PUBLIC;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA safepath TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO PUBLIC;

-- Create test database objects (for development/testing)
DO $$
BEGIN
    IF current_database() LIKE '%test%' THEN
        -- Additional test-specific setup
        INSERT INTO filter_rules (name, description, pattern, action, severity, category, priority) VALUES
            ('test_rule', 'Test rule for development', 'test_pattern', 'flag', 'low', 'test', 1)
        ON CONFLICT (name) DO NOTHING;
    END IF;
END $$;

-- Log initialization completion
INSERT INTO monitoring.health_checks (check_name, status, response_time_ms) 
VALUES ('database_initialization', 'healthy', 0);

-- Display initialization summary
DO $$
BEGIN
    RAISE NOTICE '=== CoT SafePath Filter Database Initialization Complete ===';
    RAISE NOTICE 'Schemas created: safepath, audit, monitoring';
    RAISE NOTICE 'Tables created: % in safepath schema', (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'safepath');
    RAISE NOTICE 'Default rules inserted: %', (SELECT COUNT(*) FROM filter_rules WHERE name NOT LIKE 'test_%');
    RAISE NOTICE 'System configuration entries: %', (SELECT COUNT(*) FROM system_config);
END $$;