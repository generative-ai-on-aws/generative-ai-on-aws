curl -X POST "http://localhost:80/admin/export" \
	-H "Authorization: Bearer XXX" \
	-H "Content-Type: application/json" \
	-d '{"model": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"}'

