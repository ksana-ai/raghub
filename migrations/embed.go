// Package migrations exposes the SQL migrations embedded in the raghub
// binary. Keeping the migration source at the repository root also makes it
// usable by external migration tooling.
package migrations

import "embed"

// Files contains every forward SQL migration.
//
//go:embed *.up.sql
var Files embed.FS
