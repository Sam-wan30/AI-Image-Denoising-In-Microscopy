"""Production services for the Flask web app."""

def get_denoiser_service():
	"""Lazily import the denoiser service when needed."""
	from services.denoiser import get_denoiser_service as _get_denoiser_service

	return _get_denoiser_service()

__all__ = ["get_denoiser_service"]
