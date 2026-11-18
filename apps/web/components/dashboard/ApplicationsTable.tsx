import { APPLICATION_STATUS_LABELS, type ApplicationStatus } from "@/lib/content";
import styles from "./Dashboard.module.css";

export type ApplicationRow = {
  id: string;
  company: string;
  role: string;
  status: ApplicationStatus;
  applied: string;
};

export default function ApplicationsTable({ rows }: { rows: ApplicationRow[] }) {
  return (
    <section id="applications" className={styles.card} aria-labelledby="applications-title">
      <div className={styles.cardHead}>
        <h2 id="applications-title">Applications</h2>
        <span className={styles.hint}>Shows up because you&apos;re job hunting</span>
      </div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Company</th>
              <th scope="col">Role</th>
              <th scope="col">Status</th>
              <th scope="col">Applied</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td className={styles.company}>{row.company}</td>
                <td>{row.role}</td>
                <td>
                  <span className={`${styles.status} ${styles[`status_${row.status}`]}`}>
                    {APPLICATION_STATUS_LABELS[row.status]}
                  </span>
                </td>
                <td className={styles.meta}>{row.applied}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
