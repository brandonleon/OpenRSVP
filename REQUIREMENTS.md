## Introduction:
- Brief overview of the OpenRSVP App.
- No logins, utilizing secret links with UUIDs.
## Functional Requirements:
### Event Creation:
- Ability to create events with details (name, date, etc.).
- Generate a unique secret link with a UUID for each OpenRSVP event.
### RSVP Management:
- Users can RSVP (Yes, No, Maybe) through a form on the OpenRSVP event page.
- Modify RSVP through links sent via email.
### Email Subscription:
- Users can request email notifications for OpenRSVP events they are RSVP'd to.
- Confirmation emails for successful subscriptions.
## Data Models:
### Event Table:
- Fields: Event ID (UUID), Name, Date, Secret Link, etc.
### People Table:
- Fields: User ID (UUID), Name, Email, etc.
### RSVP Table:
- Fields: RSVP ID (UUID), User ID (foreign key linking to People Table), Event ID (foreign key linking to Event Table), RSVP Status, etc.
## Database Management:
- Store user info and OpenRSVP event data for one year.
- Configurable purge time for both users and OpenRSVP events via a config file.
- No purging if a user is associated with an OpenRSVP event.
## User Interaction:
- Pages for creating OpenRSVP events, RSVP, modifying RSVP, and subscribing to OpenRSVP event emails.
- Confirmation messages for successful actions.
## Security:
- Secure generation and handling of secret links with UUIDs.
- Validation and sanitation of user inputs to prevent vulnerabilities.
## Configuration:
- Implement a configuration file to set parameters like purge time for users and OpenRSVP events.
## Documentation:
- Detailed documentation for OpenRSVP API endpoints, data models, and configuration options.
- Provide examples and use cases for better understanding.
## Testing:
- Unit tests for OpenRSVP API endpoints, ensuring data integrity and proper functionality.
## Deployment:
- Guidelines for deploying the OpenRSVP FastAPI app in production.