FROM certbot/certbot:latest
RUN apk add --no-cache docker-cli
COPY entrypoint.sh /entrypoint.sh
COPY renew.sh /renew.sh
RUN chmod +x /entrypoint.sh /renew.sh \
    && echo "0 */12 * * * /renew.sh >> /var/log/certbot-cron.log 2>&1" > /etc/crontabs/root
ENTRYPOINT ["/entrypoint.sh"]
CMD []
