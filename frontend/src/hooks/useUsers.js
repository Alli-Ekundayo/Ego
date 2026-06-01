import { useState, useEffect } from "react";

export function useUsers() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchUsers() {
      try {
        const pageSize = 100;
        const res = await fetch(`/api/users?page=1&page_size=${pageSize}`);
        if (!res.ok) throw new Error("Failed to load users");
        const data = await res.json();
        let allUsers = data.items;
        const total = data.total;

        if (allUsers.length < total) {
          const numPages = Math.ceil(total / pageSize);
          const promises = [];
          for (let i = 2; i <= numPages; i++) {
            promises.push(fetch(`/api/users?page=${i}&page_size=${pageSize}`).then(r => r.json()));
          }
          const pages = await Promise.all(promises);
          pages.forEach(page => {
            allUsers = allUsers.concat(page.items);
          });
        }

        allUsers.sort((a, b) => a.name.localeCompare(b.name));
        
        setUsers(allUsers);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    fetchUsers();
  }, []);

  return { users, loading };
}
