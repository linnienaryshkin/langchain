export const logger = {
  info: (message: string, params?: Record<string, any>) => {
    console.log(
      JSON.stringify(
        { time: new Date().toISOString(), level: "INFO", message, params },
        null,
        2
      )
    );
  },
  error: (message: string, params?: Record<string, any>) => {
    console.error(
      JSON.stringify(
        { time: new Date().toISOString(), level: "ERROR", message, params },
        null,
        2
      )
    );
  },
  debug: (message: string, params?: Record<string, any>) => {
    console.debug(
      JSON.stringify(
        { time: new Date().toISOString(), level: "DEBUG", message, params },
        null,
        2
      )
    );
  },
};
