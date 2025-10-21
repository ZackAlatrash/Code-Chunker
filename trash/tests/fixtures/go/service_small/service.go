package foreca

import (
	"context"
	"encoding/json"
	"time"

	"go.uber.org/zap"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/sync/singleflight"
	"go.impalastudios.com/weather/foreca_proxy/internal/cache"
)

// providerClient defines the interface for weather forecast providers
type providerClient interface {
	GetForecastForLocation(ctx context.Context, location string) (*Forecast, error)
}

// mappingsRepository defines the interface for location mappings
type mappingsRepository interface {
	GetMappingByID(ctx context.Context, id string) (*Mapping, error)
}

// cacheClient defines the interface for caching forecast data
type cacheClient interface {
	Get(ctx context.Context, key string) (*cache.Item, error)
	Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
}

// Service aggregates singleflight, provider, mappings, and cache clients with a TTL
type Service struct {
	provider  providerClient
	mappings  mappingsRepository
	cache     cacheClient
	sf        *singleflight.Group
	logger    *zap.Logger
	tracer    trace.Tracer
	ttl       time.Duration
}

// NewService creates a new Service instance
func NewService(
	provider providerClient,
	mappings mappingsRepository,
	cache cacheClient,
	logger *zap.Logger,
	ttl time.Duration,
) *Service {
	return &Service{
		provider: provider,
		mappings: mappings,
		cache:    cache,
		sf:       &singleflight.Group{},
		logger:   logger,
		tracer:   otel.Tracer("foreca-service"),
		ttl:      ttl,
	}
}

// GetForecastForLocation retrieves weather forecast for a location with caching and singleflight deduplication
func (s *Service) GetForecastForLocation(ctx context.Context, locationID string) (*Forecast, error) {
	ctx, span := s.tracer.Start(ctx, "GetForecastForLocation")
	defer span.End()

	// Check cache first
	cacheKey := "forecast:" + locationID
	cachedItem, err := s.cache.Get(ctx, cacheKey)
	if err == nil && cachedItem != nil {
		span.SetAttributes(attribute.Bool("cache_hit", true))
		s.logger.Info("Cache hit for location", zap.String("location_id", locationID))
		
		var forecast Forecast
		if err := json.Unmarshal(cachedItem.Data, &forecast); err != nil {
			s.logger.Error("Failed to unmarshal cached forecast", zap.Error(err))
			return nil, err
		}
		return &forecast, nil
	}

	span.SetAttributes(attribute.Bool("cache_hit", false))

	// Use singleflight to deduplicate concurrent requests
	result, err, _ := s.sf.Do(locationID, func() (interface{}, error) {
		// Load timezone for the location
		location, err := s.mappings.GetMappingByID(ctx, locationID)
		if err != nil {
			span.SetAttributes(attribute.Bool("mapping_error", true))
			return nil, err
		}

		// Get forecast from provider
		forecast, err := s.provider.GetForecastForLocation(ctx, location.Name)
		if err != nil {
			span.SetAttributes(attribute.Bool("provider_error", true))
			return nil, err
		}

		// Cache the result
		forecastData, err := json.Marshal(forecast)
		if err != nil {
			s.logger.Error("Failed to marshal forecast for caching", zap.Error(err))
		} else {
			if err := s.cache.Set(ctx, cacheKey, forecastData, s.ttl); err != nil {
				s.logger.Error("Failed to cache forecast", zap.Error(err))
			}
		}

		return forecast, nil
	})

	if err != nil {
		span.SetAttributes(attribute.Bool("error", true))
		return nil, err
	}

	return result.(*Forecast), nil
}

// Forecast represents weather forecast data
type Forecast struct {
	Location    string    `json:"location"`
	Temperature float64   `json:"temperature"`
	Humidity    int       `json:"humidity"`
	Timestamp   time.Time `json:"timestamp"`
}

// Mapping represents location mapping data
type Mapping struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}
